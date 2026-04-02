# Programming Practical 5 - Pretraining GPT2 on a cluster


## 1. Cheatsheet on TURPAN cluster

### Logging in the cluster

For convenience, create a config file in `~/.ssh/config` with the following content:

```
Host turpan
    Hostname turpanlogin.calmip.univ-toulouse.fr
    User YOUR_USERNAME
    PreferredAuthentications password
    PubkeyAuthentication no
```

Then you can simply log in with:

```bash
ssh turpan
```

### Updating the repository

To update your forked repo with the latest changes from this original repo, run:

```bash
git fetch upstream
git merge upstream/main
```

### Work remotely from vscode

I strongly encourage you to use vscode remote environment to work on the project. On the leftbar of vscode, you should see an icon "Remote Explorer". Click on it, then click on "SSH" if needed, and click the left arrow next to "turpan". You will have to fill your password. Once you are connected, you can open the project folder and work on it as if it was local. You can even run the code in a terminal in vscode.

### `uv` environment and AppTainer

First, create an env directory in `/tmpdir`:

```bash
mkdir -p /tmpdir/YOUR_USERNAME/envs
```

Save the following command as an alias in your `~/.bashrc` to avoid having to write it every time. Add these lines 5DO NOT FORGET TO REPLACE `YOUR_USERNAME`:

```bash
alias run_apptainer_login="apptainer shell \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai \
--env HF_HOME=/work/shared/TPIRT/huggingface \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif"
```
and then refresh the `~/.bashrc` with `source ~/.bashrc`. Then run 
```bash
run_apptainer_login
```
to launch the apptainer image on a login node.

You are now in the `apptainer` image! Install the env using:

```bash
uv venv --system-site-packages /tmpdir/YOUR_USERNAME/envs/aai
uv sync --only-group turpan
``` 
Now the environment should be up and running.

## Running the code on the compute nodes

To run some code on the compute nodes, you have two choices. You can either use the node in interactive mode, meaning that you have a shell on the compute node where you can run commands one by one, or you can submit a job, meaning that you write a script with the commands you want to run and submit it to the cluster, which will run it for you.

### Interactive mode

First, add this alias in your `~/.bashrc` to avoid having to write it every time. Add these lines (DO NOT FORGET TO REPLACE `YOUR_USERNAME`):

```bash
alias run_apptainer_gpu="srun -p shared -n1 --gres=gpu:1 --pty apptainer shell \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai \
--env UV_NO_SYNC=true \
--env HF_HOME=/work/shared/TPIRT/huggingface \
--env HF_HUB_OFFLINE=1 \
--env HF_DATASETS_OFFLINE=1 \
--env TRANSFORMERS_OFFLINE=1 \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif"
```
and then refresh the `~/.bashrc` with `source ~/.bashrc`. Then run 

```bash
run_apptainer_gpu
```

to launch the apptainer image on a compute node.

You can check that you are on a compute node by using `nvidia-smi`. Then you can run your scripts as you would do in a local machine.

### Batch mode

For long scirpts, often running overnight, you do not want to keep your terminal open. Instead, you will set up an instruction script (**a job**) giving the cluster all the information it needs to run your code. This script is an `.sbatch` script and looks like this:

```bash
#!/bin/bash
#SBATCH -J mon_job
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p shared
#SBATCH --time=00:15:00 
#SBATCH --reservation=tpirt4
#SBATCH --output=/users/formation/YOUR_USERNAME/job_results/out/job_%j.out
#SBATCH --error=/users/formation/YOUR_USERNAME/job_results/err/job_%j.err


apptainer exec \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai \
--env UV_NO_SYNC=true \
--env HF_HOME=/work/shared/TPIRT/huggingface \
--env HF_HUB_OFFLINE=1 \
--env HF_DATASETS_OFFLINE=1 \
--env TRANSFORMERS_OFFLINE=1 \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
uv run python mon_script.py
```

You can find this tamplate on `sbatch_scripts/template.sbatch` in the project. Take the time to understand each `#SBATCH` line of the script:

- `--nodes 1`: Number of nodes to use (1 in our case)
- `--ntasks 1`: Number of tasks to run (1 in our case, it is the number of times the command will be run.
- `--cpus-per-task=8`: Number of CPU cores to allocate for this job (8 in our case, change it according to your needs)
- `--gres=gpu:1`: Number of GPU to use (1 in our case)
- `-p shared`: Partition to use (shared in our case, do not change this, it tells the cluster not to use the full node)
- `--time=00:15:00`: Time limit for the job (15 minutes in this case, change it according to your needs)
- `--reservation=tpirt4`: Reservation to use (tpirt4 in this case, change it according to the schedule of the PP sessions - see below)
- `--output`: Path to the file where the standard output of the job will be saved (change YOUR_USERNAME and the path according to your needs)
- `--error`: Path to the file where the standard error of the job will be saved (change YOUR_USERNAME and the path according to your needs)

**Replace `YOUR_USERNAME` and `mon_script.py` with your username and the script you want to run.**

Let's call this file `run_job.sbatch`. You can submit this job to the cluster with:

```bash
sbatch --reservation=tpirt4 run_job.sbatch
```

and check the status of your job with:

```bash
jobinfo job_id
```
where `job_id` is the id of your job given by the output of the `sbatch` command. You can also check the id using:

```bash
squeue -u $USER -l
```

which displays informations about runing jobs.


!!! Note
    The `--reservation=tpirt4` option is specific to this cluster and allows you to use the reserved resources for the Programming Practical sessions. The reference `tpirt4` will change for each PP following this schedule:


```
tpirt1 2026-03-09 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt2 2026-03-13 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt4 2026-03-20 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt5 2026-04-03 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt6 2026-05-27 10:30:00 - 16:00:00 (Duree : 05 H)
tpirt7 2026-05-29 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt8 2026-06-01 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt9 2026-06-03 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt10 2026-06-05 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt11 2026-06-08 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt12 2026-06-10 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt13 2026-06-12 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt14 2026-06-19 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt15 2026-06-22 08:00:00 - 10:00:00 (Duree : 02 H)
tpirt16 2026-06-23 10:00:00 - 12:00:00 (Duree : 02 H)
tpirt17 2026-06-24 14:00:00 - 16:00:00 (Duree : 02 H)
tpirt18 2026-06-26 08:00:00 - 10:00:00 (Duree : 02 H)
```

The code will run silently on the cluster but it will output `stdin` and `stderr` in `--output` and `--error` paths specified in the `.sbatch` script. First, create the paths:

```bash
mkdir -p ~/job_results/out
mkdir -p ~/job_results/err
```

Then you can check the output and error of your job with (Replace job_id with your job_id):

```bash
cat ~/job_results/out/job_job_id.out
cat ~/job_results/err/job_job_id.err
```

Or open it in vscode and refresh it when you want to check the output / error. One super convenient way to check the output on vscode is to click File > Add folder to workspace, then add the `job_results` folder. Then you can open the `out` and `err` folders in the vscode explorer alongside your code and open the output and error files of your job.


## 5. Warmup on this semester project: easy VLM.

As a warmup for this semester's project, you will launch a dummy training of a small VLM on [Flickr30k dataset](https://huggingface.co/datasets/AnyModal/flickr30k) after completing the missing parts of the modality projector and the "glue" code that combines the text and image features. The goal of this warmup is to get familiar with the model, the training loop, and the cluster environment before launching the real training for the project.

### Complete the missing parts

Before launching your first training on the cluster, you will have to complete the missing parts in the code involved in training and generation, as in previous PPs. Start by reading `model.py` and `train.py` to understand the overall structure.

---

#### `get_batch` in `train.py`

**What it does:** a simple data loader that samples random chunks from a memory-mapped binary file of token ids. Each call returns a batch of input-target pairs for next-token prediction.

The data is stored as a flat array of token ids. For a language model, the input `x` is a window of `block_size` consecutive tokens, and the target `y` is the same window shifted by one position (i.e., for each token in `x`, the target is the next token in the sequence).

```python
def get_batch(split):
    ...
    data = np.memmap(...)
    ix = # TODO: sample batch_size random starting indices in [0, len(data) - block_size)
    x  = # TODO: for each index i in ix, extract a chunk of block_size tokens as input
    y  = # TODO: for each index i in ix, extract a chunk of block_size tokens shifted by 1 as target
    ...
```

- `ix`: use `torch.randint` to sample `batch_size` random starting positions. The upper bound should ensure that a full window of `block_size` tokens fits.
- `x`: for each starting index `i`, slice between `i` and `i + block_size`, convert to `np.int64`, wrap in a tensor, and stack all slices into a `(batch_size, block_size)` tensor.
- `y`: same as `x`, but shifted by one.

---

#### Training loop in `train.py`

**What it does:** the forward-backward pass inside the gradient accumulation loop. The model receives input tokens `X` and must produce predictions that are compared against the targets `Y` using cross-entropy loss.

```python
for micro_step in range(gradient_accumulation_steps):
    with ctx:
        logits = # TODO: forward pass through the model
        loss   = # TODO: compute the cross-entropy loss
        loss = loss / gradient_accumulation_steps
    ...
```

- `logits`: call the model on `X`. The output has shape `(batch_size, block_size, vocab_size)`.
- `loss`: use `F.cross_entropy`. You need to reshape `logits` to `(B*T, vocab_size)` and `Y` to `(B*T)`. Use `ignore_index=-1`.

---

#### `generate` in `model.py`

**What it does:** autoregressive text generation. Given a conditioning sequence of token ids, the model predicts one token at a time, appends it to the sequence, and repeats.

The generation loop has the following steps:

1. **Forward pass** — run the (possibly cropped) context through the model to get logits.
2. **Select & scale** — take only the logits at the last time step and divide by the temperature.
3. **Top-k filtering** — optionally zero out all logits outside the top-k most likely tokens.
4. **Sample** — convert logits to probabilities with softmax, then sample one token.
5. **Append** — concatenate the new token to the running sequence.

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits = # TODO: call the model on idx_cond
        logits = # TODO: select the logits at the last time step (index -1), divide by temperature
        if top_k is not None:
            v, _ = # TODO: use torch.topk to get the top-k values
            # TODO: set all logits below the smallest top-k value (v[:, [-1]]) to -float('Inf')
        probs    = # TODO: apply softmax to get probabilities
        idx_next = # TODO: sample from the distribution (torch.multinomial)
        idx      = # TODO: concatenate idx_next to idx along dim 1
    return idx
```

- For the forward pass, simply call the model — the model's `forward` method returns logits of shape `(batch, seq_len, vocab_size)`.
- `logits[:, -1, :]` selects the predictions for the last position. Dividing by `temperature` controls randomness: lower temperature → more deterministic.
- `torch.topk` returns the `k` largest values; use the smallest of those as a threshold to mask everything else to `-Inf`.
- `F.softmax(logits, dim=-1)` converts to probabilities; `torch.multinomial(probs, num_samples=1)` draws one sample per row.
- `torch.cat((idx, idx_next), dim=1)` grows the sequence by one token.


### Training on Flickr30k



