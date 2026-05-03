/**
 * Worker/queue summary API.
 *
 * Legacy `/api/v1/jobs/stats` was removed with the analysis job system.
 * Frontend callers still expect a compact queue summary shape, so this module
 * derives one from `/api/v1/analysis-jobs`.
 */

import { listAnalysisJobs } from './analysis';

function sumFailed(failed) {
  if (!failed || typeof failed !== 'object') return 0;
  return Object.values(failed).reduce((acc, value) => acc + (value || 0), 0);
}

export async function getJobStats() {
  const data = await listAnalysisJobs(true);
  const jobs = (data?.jobs || []).filter((job) => job.status !== 'cancelled' && job.status !== 'archived');

  const summary = {
    total: 0,
    total_files: 0,
    complete_files: 0,
    completed: 0,
    failed: 0,
    pending: 0,
    assigned: 0,
    processing: 0,
    throughput: 0,
    eta_seconds: null,
    download_waiting: 0,
    ready_pending: 0,
    phase_parse_done: 0,
    phase_vision_done: 0,
    phase_embed_done: 0,
  };

  let maxEta = null;

  for (const job of jobs) {
    const progress = job.progress || {};
    const phases = progress.phases || {};
    const failedMap = progress.failed || {};
    const total = progress.total || 0;
    const complete = progress.complete || 0;
    const failed = sumFailed(failedMap);
    const remaining = Math.max(total - complete - failed, 0);

    summary.total += total;
    summary.total_files += total;
    summary.complete_files += complete;
    summary.completed += complete;
    summary.failed += failed;
    summary.pending += remaining;
    summary.download_waiting += phases.download || 0;
    summary.ready_pending += phases.parse || 0;
    summary.phase_parse_done += progress.parsed || 0;
    summary.phase_vision_done += progress.mc_done || 0;
    summary.phase_embed_done += progress.mv_done || 0;

    if (job.status === 'active') {
      summary.processing += (phases.ai || 0) + (phases.mv || 0);
    } else if (job.status === 'paused') {
      summary.assigned += (phases.ai || 0) + (phases.mv || 0);
    }

    const metrics = job.metrics || {};
    const throughput = metrics.throughput?.files_per_min || 0;
    const etaSeconds = metrics.throughput?.eta_seconds;
    summary.throughput = Math.max(summary.throughput, throughput);
    if (etaSeconds != null) {
      maxEta = maxEta == null ? etaSeconds : Math.max(maxEta, etaSeconds);
    }
  }

  summary.eta_seconds = maxEta;
  return summary;
}
