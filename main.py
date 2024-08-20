import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.DataFrame({
    'Name': ['FRANKENSTEIN', 'NCI1', 'COIL-RAG', 'Letter-high', 'DD', 'PROTEINS_full', 'COLORS-3'],
    'Ave. generalization error': [-0.0015, 0.0479, -0.0043, 0.0513, 0.1483, -0.0213, 0.0090]
})

df2 = pd.DataFrame({
    'Name': ['FRANKENSTEIN', 'NCI1', 'COIL-RAG', 'Letter-high', 'DD', 'PROTEINS_full', 'COLORS-3'],
    'Ave. degree': [2.0627741588132293, 2.1550138021472596, 1.8277218368573067, 1.8896836075252956, 4.979061666490266, 3.7346421664401634, 2.928853359699249],
    'Ave. shortest path': [3.6521139346083733, 5.469777618361717, 1.0606131260794474, 1.5173343777117363, 7.970076470113258, 4.712380854361992, 2.0945741756348464]
})

df_combined = pd.merge(df1, df2, on='Name')

df_sorted_by_degree = df_combined.sort_values('Ave. degree')
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(df_sorted_by_degree['Ave. degree'], df_sorted_by_degree['Ave. generalization error'], marker='o', color='b')
plt.title('Ave. Degree vs. Ave. Generalization Error')
plt.xlabel('Ave. Degree')
plt.ylabel('Ave. Generalization Error')

df_sorted_by_shortest_path = df_combined.sort_values('Ave. shortest path')
plt.subplot(1, 2, 2)
plt.plot(df_sorted_by_shortest_path['Ave. shortest path'], df_sorted_by_shortest_path['Ave. generalization error'], marker='o', color='r')
plt.title('Ave. Shortest Path vs. Ave. Generalization Error')
plt.xlabel('Ave. Shortest Path')
plt.ylabel('Ave. Generalization Error')

plt.tight_layout()
plt.show()
