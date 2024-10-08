python3 -m debugpy --listen 5678 --wait-for-client  autoregressive/sample/sample_t2i.py --cfg-scale 7.5 --top-k 5000 --no-left-padding
# python autoregressive/sample/sample_t2i.py --cfg-scale 7.5 --top-k 5000
#!/bin/bash  

# Iterate over cfg-scale values from 1 to 7.5 with a step of 0.5  
# for cfg_scale in $(seq 1 0.5 7.5)  
# do  
#   # Iterate over top-k values  
#   for top_k in 1 10 100 200 500 1000 2000 3000 4000 5000 10000 16384  
#   do  
#     # Run the Python command with the current cfg-scale and top-k values  
#     python autoregressive/sample/sample_t2i.py --cfg-scale "$cfg_scale" --top-k "$top_k"  
#   done  
# done