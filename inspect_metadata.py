import pickle
with open('data/vector_store/index.pkl', 'rb') as f:
    data = pickle.load(f)

docs = data['documents']
seen_types = set()
for d in docs:
    dtype = d.metadata.get('document_type')
    if dtype not in seen_types:
        seen_types.add(dtype)
        print('---', dtype, '---')
        print(d.metadata)
        print()