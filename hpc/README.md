Get access via ssh
`ssh -i ~/.ssh/id_rsa_palma s_deis02@palma.uni-muenster.de`

Start interactive session on GPU
`srun --nodes 1 -t 00:30:00 --partition gpuk20 --pty bash`

Load modules

```
module load palma/2019b
module load fosscuda/2019b
module load PyTorch/1.4.0-Python-3.7.4
```