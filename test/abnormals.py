import pickle
with open('resources/prev_clean/sample_20/qwen/prev_clean_prev_window_1_no_prob/prev_to_abnormals.pkl', 'rb') as f:
    data = pickle.load(f)
    print(len(data))          # 应该 > 0
    print(list(data.keys())[:5])  # 查看几个前置词