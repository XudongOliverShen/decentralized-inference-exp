python /root/work/decentralized-inference-exp/eval_ppl.py --first_k_tokens 0 --load_in_8bit --result_dir ./results/sanity_checks
python /root/work/decentralized-inference-exp/eval_ppl.py --first_k_tokens 100000 --load_in_8bit --result_dir ./results/sanity_checks
python /root/work/decentralized-inference-exp/eval_ppl.py --first_k_tokens 50000 --load_in_8bit --result_dir ./results/sanity_checks
python /root/work/decentralized-inference-exp/eval_ppl.py --first_k_tokens 20000 --load_in_8bit --result_dir ./results/sanity_checks
# python /root/work/decentralized-inference-exp/eval_ppl.py --first_k_tokens 10000 --load_in_8bit --result_dir ./results/sanity_checks