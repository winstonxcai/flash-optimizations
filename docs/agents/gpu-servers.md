# GPU servers and shared VDI storage

The VDI has separate VS Code Remote SSH windows for two GPU nodes:

| SSH host | Hardware |
|---|---|
| `10.72.1.171` | 8 × NVIDIA H100 80 GB |
| `10.72.1.175` | 8 × NVIDIA H100 80 GB |

- Confirm the VS Code SSH host and terminal hostname before submitting jobs,
  inspecting processes or stopping servers. GPUQ queues, job status, GPU IDs
  and running processes belong to that node; inspect both nodes when deciding
  where an authorized job can run. Request GPU counts through GPUQ and use its
  assigned devices rather than manually choosing GPU IDs.
- Both nodes access the same shared disk for
  `/home/jovyan/winstonxcai/flash-optimizations` and its repository folders,
  including source, data, logs, results and transfer packets. Files do not need
  copying between these nodes. Git operations, source edits and permission
  changes on one node affect the other. Do not pull, change fork pins or edit
  shared serving sources while a job on either node depends on that checkout.
- Check existing images and containers on the destination node first. Matching
  image/container names and shared repository files do not establish matching
  Docker image IDs or container state. Reuse an existing matching image; do not
  export, rebuild or recreate an image merely because execution moves nodes.
  If the recorded image ID differs, establish the intended baseline before
  aligning it. Preserve unrelated running containers and servers.
- Record the execution node alongside each GPUQ job ID and assigned GPUs. When
  moving a job, confirm cancellation on the original node, stop its old result
  watcher and submit the replacement once. Keep cancelled-job evidence and
  prevent collectors on different nodes from publishing the same run twice.
- GPUQ may execute as `gpuq-runner` (group `gpuq`) rather than the interactive
  root account. Verify that this identity can access prepared inputs and write
  the run's artifacts before queueing. Let GPUQ manage its job logs; use the
  existing serving/replay artifact paths rather than extra logging scaffolding.
