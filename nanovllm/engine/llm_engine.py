import atexit
from dataclasses import dataclass, fields
from math import isfinite
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine.block_manager import PrefixCachePreview
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import MixedScheduleBatch, Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.pd_disaggregation import restore_decode_sequence


@dataclass(frozen=True, slots=True)
class SequenceStepEvent:
    seq_id: int
    phase: str
    num_scheduled_tokens: int
    emitted_token_id: int | None
    finished: bool
    prefill_kind: str | None = None
    actual_recompute_tokens: int = 0
    resumed: bool = False
    scheduler_step_index: int | None = None
    previous_progress_step: int | None = None
    progress_gap_steps: int | None = None
    had_emitted_token_before: bool = False


@dataclass(frozen=True, slots=True)
class EngineStepResult:
    outputs: list[tuple[int, list[int]]]
    phase: str
    num_scheduled_tokens: int
    signed_token_count: int
    events: list[SequenceStepEvent]
    phase_timings_ms: dict[str, float] | None = None
    scheduler_event: object | None = None


class LLMEngine:

    def __init__(self, model, **kwargs):
        self._exited = False
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.cuda_event_timing = config.cuda_event_timing
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def configure_request_slo(
        self,
        seq_id: int,
        *,
        ttft_deadline_at: float,
        itl_slo_s: float,
    ) -> None:
        if self.scheduler.policy not in {
            "recompute_aware_bounded",
            "mixed_slo_budget",
        }:
            return
        if (
            not isinstance(ttft_deadline_at, (int, float))
            or isinstance(ttft_deadline_at, bool)
            or not isfinite(ttft_deadline_at)
        ):
            raise ValueError("ttft_deadline_at must be finite")
        if (
            not isinstance(itl_slo_s, (int, float))
            or isinstance(itl_slo_s, bool)
            or not isfinite(itl_slo_s)
            or itl_slo_s <= 0
        ):
            raise ValueError("itl_slo_s must be positive")
        matches = [
            seq
            for seq in [
                *self.scheduler.running,
                *self.scheduler.waiting,
            ]
            if seq.seq_id == seq_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Cannot configure SLO for unknown sequence: {seq_id}"
            )
        seq = matches[0]
        seq.ttft_deadline_at = ttft_deadline_at
        seq.itl_slo_s = itl_slo_s

    def preview_prefix_cache(
        self,
        prompt: list[int],
        max_tokens: int,
    ) -> PrefixCachePreview:
        return self.scheduler.block_manager.preview_prefix_cache(
            prompt,
            max_tokens,
        )

    def cache_state_snapshot(self) -> list[dict[str, object]]:
        return self.scheduler.block_manager.cache_state_snapshot()

    def step(self):
        result = self.step_with_events()
        return result.outputs, result.signed_token_count

    def step_with_events(self) -> EngineStepResult:
        step_started = perf_counter()
        schedule_started = perf_counter()
        scheduled = self.scheduler.schedule()
        if isinstance(scheduled, MixedScheduleBatch):
            return self._step_mixed_with_events(
                scheduled,
                step_started=step_started,
                schedule_started=schedule_started,
            )
        seqs, is_prefill = scheduled
        schedule_cpu_ms = (perf_counter() - schedule_started) * 1000
        scheduled_tokens = [seq.num_scheduled_tokens for seq in seqs]
        completion_lengths = [seq.num_completion_tokens for seq in seqs]
        previous_progress_steps = [seq.last_progress_step for seq in seqs]
        had_emitted_tokens = [
            seq.num_completion_tokens > 0 for seq in seqs
        ]
        recompute_flags = [
            bool(is_prefill and seq.pending_recompute) for seq in seqs
        ]
        resume_counts = [seq.resume_count for seq in seqs]
        num_scheduled_tokens = sum(scheduled_tokens)
        signed_token_count = (
            num_scheduled_tokens if is_prefill else -num_scheduled_tokens
        )
        runner_start = runner_end = None
        cuda_event_timing = getattr(self, "cuda_event_timing", False)
        if cuda_event_timing and torch.cuda.is_available():
            runner_start = torch.cuda.Event(enable_timing=True)
            runner_end = torch.cuda.Event(enable_timing=True)
            runner_start.record()
        runner_wall_started = perf_counter()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        runner_wall_ms = (perf_counter() - runner_wall_started) * 1000
        if runner_start is not None and runner_end is not None:
            runner_end.record()
            runner_end.synchronize()
            model_runner_cuda_ms = runner_start.elapsed_time(runner_end)
        else:
            model_runner_cuda_ms = None
        postprocess_started = perf_counter()
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        postprocess_cpu_ms = (perf_counter() - postprocess_started) * 1000
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        phase = "prefill" if is_prefill else "decode"
        events = [
            SequenceStepEvent(
                seq_id=seq.seq_id,
                phase=phase,
                num_scheduled_tokens=scheduled,
                emitted_token_id=(
                    token_id
                    if seq.num_completion_tokens > completion_length
                    else None
                ),
                finished=seq.is_finished,
                prefill_kind=(
                    "recompute_prefill"
                    if is_recompute
                    else "initial_prefill"
                    if is_prefill
                    else None
                ),
                actual_recompute_tokens=(
                    scheduled if is_recompute else 0
                ),
                resumed=(seq.resume_count > resume_count),
                scheduler_step_index=getattr(
                    self.scheduler,
                    "_scheduler_step_index",
                    None,
                ),
                previous_progress_step=previous_progress_step,
                progress_gap_steps=seq.last_progress_gap_steps,
                had_emitted_token_before=had_emitted_token,
            )
            for (
                seq,
                scheduled,
                completion_length,
                token_id,
                is_recompute,
                resume_count,
                previous_progress_step,
                had_emitted_token,
            ) in zip(
                seqs,
                scheduled_tokens,
                completion_lengths,
                token_ids,
                recompute_flags,
                resume_counts,
                previous_progress_steps,
                had_emitted_tokens,
                strict=True,
            )
        ]
        phase_timings_ms = None
        if cuda_event_timing:
            phase_timings_ms = {
                "schedule_cpu_ms": schedule_cpu_ms,
                "model_runner_wall_ms": runner_wall_ms,
                "postprocess_cpu_ms": postprocess_cpu_ms,
                "step_wall_ms": (perf_counter() - step_started) * 1000,
            }
            if model_runner_cuda_ms is not None:
                phase_timings_ms["model_runner_cuda_ms"] = model_runner_cuda_ms
        return EngineStepResult(
            outputs=outputs,
            phase=phase,
            num_scheduled_tokens=num_scheduled_tokens,
            signed_token_count=signed_token_count,
            events=events,
            phase_timings_ms=phase_timings_ms,
            scheduler_event=self.scheduler.last_step_event,
        )

    def _step_mixed_with_events(
        self,
        batch: MixedScheduleBatch,
        *,
        step_started: float,
        schedule_started: float,
    ) -> EngineStepResult:
        schedule_cpu_ms = (perf_counter() - schedule_started) * 1000
        completion_lengths = {
            item.sequence.seq_id: item.sequence.num_completion_tokens
            for item in batch.items
        }
        recompute_flags = {
            item.sequence.seq_id: (
                item.phase == "prefill" and item.sequence.pending_recompute
            )
            for item in batch.items
        }
        resume_counts = {
            item.sequence.seq_id: item.sequence.resume_count
            for item in batch.items
        }
        runner_started = perf_counter()
        sampled = self.model_runner.call("run_mixed", batch)
        runner_wall_ms = (perf_counter() - runner_started) * 1000
        postprocess_started = perf_counter()
        self.scheduler.postprocess_mixed(batch, sampled)
        postprocess_cpu_ms = (perf_counter() - postprocess_started) * 1000
        phases = {item.phase for item in batch.items}
        phase = "mixed" if len(phases) > 1 else next(iter(phases))
        events = []
        outputs = []
        for item in batch.items:
            seq = item.sequence
            token_id = sampled.get(seq.seq_id)
            events.append(
                SequenceStepEvent(
                    seq_id=seq.seq_id,
                    phase=item.phase,
                    num_scheduled_tokens=item.num_scheduled_tokens,
                    emitted_token_id=(
                        token_id
                        if seq.num_completion_tokens
                        > completion_lengths[seq.seq_id]
                        else None
                    ),
                    finished=seq.is_finished,
                    prefill_kind=(
                        "recompute_prefill"
                        if recompute_flags[seq.seq_id]
                        else "initial_prefill"
                        if item.phase == "prefill"
                        else None
                    ),
                    actual_recompute_tokens=(
                        item.num_scheduled_tokens
                        if recompute_flags[seq.seq_id]
                        else 0
                    ),
                    resumed=(
                        seq.resume_count > resume_counts[seq.seq_id]
                    ),
                    scheduler_step_index=getattr(
                        self.scheduler,
                        "_scheduler_step_index",
                        None,
                    ),
                )
            )
            if seq.is_finished:
                outputs.append((seq.seq_id, seq.completion_token_ids))
        phase_timings_ms = None
        if getattr(self, "cuda_event_timing", False):
            phase_timings_ms = {
                "schedule_cpu_ms": schedule_cpu_ms,
                "model_runner_wall_ms": runner_wall_ms,
                "postprocess_cpu_ms": postprocess_cpu_ms,
                "step_wall_ms": (perf_counter() - step_started) * 1000,
            }
        return EngineStepResult(
            outputs=outputs,
            phase=phase,
            num_scheduled_tokens=batch.num_scheduled_tokens,
            signed_token_count=0 if phase == "mixed" else (
                batch.num_scheduled_tokens if phase == "prefill"
                else -batch.num_scheduled_tokens
            ),
            events=events,
            phase_timings_ms=phase_timings_ms,
            scheduler_event=self.scheduler.last_step_event,
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def reset_observability(self):
        self.scheduler.reset_observability()

    def observability_snapshot(self) -> dict:
        return self.scheduler.observability_snapshot()

    def replica_state_snapshot(self, replica_id: int) -> dict[str, int | float]:
        kv = self.scheduler.block_manager.observability_snapshot()
        return {
            "replica_id": replica_id,
            "total_kv_blocks": kv["total_blocks"],
            "free_kv_blocks": kv["final_free_blocks"],
            "used_kv_blocks": kv["final_used_blocks"],
            "kv_utilization": self.scheduler.block_manager.current_utilization(),
            "waiting_queue_length": len(self.scheduler.waiting),
            "running_requests": len(self.scheduler.running),
            "oldest_waiting_age": max(
                getattr(self.scheduler, "_waiting_age_steps", {}).values(),
                default=0,
            ),
        }

    def prefill_export(
        self,
        prompt: list[int],
        sampling_params: SamplingParams,
    ) -> dict:
        prefill_started = perf_counter()
        seq_id = self.add_request(prompt, sampling_params)
        while True:
            result = self.step_with_events()
            event = next((item for item in result.events if item.seq_id == seq_id), None)
            if event is not None and event.emitted_token_id is not None:
                break
        seq = next(item for item in self.scheduler.running if item.seq_id == seq_id)
        prefill_compute_ms = (perf_counter() - prefill_started) * 1000
        payload = self.model_runner.call("export_kv", seq, len(prompt))
        metadata = {
            "prompt_token_ids": list(prompt),
            "first_token_id": seq.completion_token_ids[0],
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "ignore_eos": sampling_params.ignore_eos,
            "materialized_tokens": len(prompt),
            "source_block_ids": list(seq.block_table),
        }
        self.scheduler.running.remove(seq)
        self.scheduler.block_manager.deallocate(seq)
        return {
            "metadata": metadata,
            "kv": payload,
            "prefill_compute_ms": prefill_compute_ms,
        }

    def import_decode(self, transfer: dict) -> int:
        metadata = transfer["metadata"]
        seq = restore_decode_sequence(
            prompt_token_ids=metadata["prompt_token_ids"],
            first_token_id=metadata["first_token_id"],
            sampling_params=SamplingParams(
                temperature=metadata["temperature"],
                max_tokens=metadata["max_tokens"],
                ignore_eos=metadata["ignore_eos"],
            ),
        )
        cached = self.scheduler.block_manager.can_allocate(
            seq, metadata["materialized_tokens"]
        )
        if cached == -1:
            raise RuntimeError("Decode worker has insufficient KV capacity")
        self.scheduler.block_manager.allocate(
            seq, 0, metadata["materialized_tokens"]
        )
        timing = self.model_runner.call("import_kv", seq, transfer["kv"])
        self.scheduler.running.append(seq)
        return {"seq_id": seq.seq_id, **timing}

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
