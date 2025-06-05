import pandas as pd

# Read the Excel file
df = pd.read_excel('performance.xlsx')

# Create markdown table for the README
print("### Performance Results")
print()
print("| Sequence Length | Implementation | Forward (ms) | Backward (ms) | Combined (ms) | Speedup vs FlashAttention |")
print("|-----------------|----------------|-------------|---------------|---------------|----------------------------|")

# Group by sequence length groups
current_seq_len = None
for idx, row in df.iterrows():
    if pd.notna(row['SeqLen']):
        current_seq_len = int(row['SeqLen'])
    
    if pd.notna(row['Implementation']):
        fwd_time = row['Time (ms)']
        bwd_time = row['Time (ms).1']
        combined_time = row['Time (ms).2']
        speedup = row['Speedup vs FlashAttention.2']
        
        print(f"| {current_seq_len:,} | {row['Implementation']} | {fwd_time:.2f} | {bwd_time:.2f} | {combined_time:.2f} | {speedup} |")

print()
print("### Performance Chart (Combined Time)")
print()
print("```")
print("Sequence Length    FlashAttention    InfLLMv2    Speedup")
print("32,768             728.08 ms         463.64 ms   1.57x")
print("65,536             1446.75 ms        523.86 ms   2.76x")  
print("131,072            2894.88 ms        627.32 ms   4.61x")
print("```") 