dblp_ct1
```shell
Processing...
Traceback (most recent call last):
  File "/home/wens/Projects/GNN/main.py", line 193, in <module>
    handle_option(args.function)
  File "/home/wens/Projects/GNN/main.py", line 174, in handle_option
    calcualte_parameters()
  File "/home/wens/Projects/GNN/main.py", line 87, in calcualte_parameters
    get_average_degree(dataset, args.verbose),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wens/Projects/GNN/utils.py", line 81, in get_average_degree
    dataset = TUDataset(root='data/TUDataset',
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 129, in __init__
    super().__init__(root, transform, pre_transform, pre_filter,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/in_memory_dataset.py", line 81, in __init__
    super().__init__(root, transform, pre_transform, pre_filter, log,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 115, in __init__
    self._process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 260, in _process
    self.process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 203, in process
    self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 36, in read_tu_data
    node_label = read_file(folder, prefix, 'node_labels', torch.long)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 100, in read_file
    return read_txt_array(path, sep=',', dtype=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 33, in read_txt_array
    return parse_txt_array(src, sep, start, end, dtype, device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 19, in parse_txt_array
    return torch.tensor([[to_number(x) for x in line.split(sep)[start:end]]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 2 at dim 1 (got 4)
```

---
dblp_ct2
```sh
Processing...
Traceback (most recent call last):
  File "/home/wens/Projects/GNN/main.py", line 193, in <module>
    handle_option(args.function)
  File "/home/wens/Projects/GNN/main.py", line 174, in handle_option
    calcualte_parameters()
  File "/home/wens/Projects/GNN/main.py", line 87, in calcualte_parameters
    get_average_degree(dataset, args.verbose),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wens/Projects/GNN/utils.py", line 81, in get_average_degree
    dataset = TUDataset(root='data/TUDataset',
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 129, in __init__
    super().__init__(root, transform, pre_transform, pre_filter,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/in_memory_dataset.py", line 81, in __init__
    super().__init__(root, transform, pre_transform, pre_filter, log,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 115, in __init__
    self._process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 260, in _process
    self.process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 203, in process
    self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 36, in read_tu_data
    node_label = read_file(folder, prefix, 'node_labels', torch.long)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 100, in read_file
    return read_txt_array(path, sep=',', dtype=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 33, in read_txt_array
    return parse_txt_array(src, sep, start, end, dtype, device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 19, in parse_txt_array
    return torch.tensor([[to_number(x) for x in line.split(sep)[start:end]]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 2 at dim 1 (got 4)
make: *** [Makefile:28: t] Error 1
```
---

DBLP_v1
```sh
Processing...
Traceback (most recent call last):
  File "/home/wens/Projects/GNN/main.py", line 193, in <module>
    handle_option(args.function)
  File "/home/wens/Projects/GNN/main.py", line 174, in handle_option
    calcualte_parameters()
  File "/home/wens/Projects/GNN/main.py", line 87, in calcualte_parameters
    get_average_degree(dataset, args.verbose),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wens/Projects/GNN/utils.py", line 81, in get_average_degree
    dataset = TUDataset(root='data/TUDataset',
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 129, in __init__
    super().__init__(root, transform, pre_transform, pre_filter,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/in_memory_dataset.py", line 81, in __init__
    super().__init__(root, transform, pre_transform, pre_filter, log,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 115, in __init__
    self._process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 260, in _process
    self.process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 203, in process
    self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 41, in read_tu_data
    node_labels = [one_hot(x) for x in node_labels]
                   ^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/utils/_one_hot.py", line 36, in one_hot
    out = torch.zeros((index.size(0), num_classes), dtype=dtype,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [enforce fail at alloc_cpu.cpp:117] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 33713596200 bytes. Error code 12 (Cannot allocate memory)
make: *** [Makefile:28: t] Error 1
```

---

facebook_ct2 
```sh
Processing...
Traceback (most recent call last):
  File "/home/wens/Projects/GNN/main.py", line 193, in <module>
    handle_option(args.function)
  File "/home/wens/Projects/GNN/main.py", line 174, in handle_option
    calcualte_parameters()
  File "/home/wens/Projects/GNN/main.py", line 87, in calcualte_parameters
    get_average_degree(dataset, args.verbose),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wens/Projects/GNN/utils.py", line 81, in get_average_degree
    dataset = TUDataset(root='data/TUDataset',
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 129, in __init__
    super().__init__(root, transform, pre_transform, pre_filter,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/in_memory_dataset.py", line 81, in __init__
    super().__init__(root, transform, pre_transform, pre_filter, log,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 115, in __init__
    self._process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 260, in _process
    self.process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 203, in process
    self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 36, in read_tu_data
    node_label = read_file(folder, prefix, 'node_labels', torch.long)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 100, in read_file
    return read_txt_array(path, sep=',', dtype=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 33, in read_txt_array
    return parse_txt_array(src, sep, start, end, dtype, device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 19, in parse_txt_array
    return torch.tensor([[to_number(x) for x in line.split(sep)[start:end]]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 2 at dim 1 (got 4)
```

---

highschool_ct1
```sh
Processing...
Traceback (most recent call last):
  File "/home/wens/Projects/GNN/main.py", line 193, in <module>
    handle_option(args.function)
  File "/home/wens/Projects/GNN/main.py", line 174, in handle_option
    calcualte_parameters()
  File "/home/wens/Projects/GNN/main.py", line 87, in calcualte_parameters
    get_average_degree(dataset, args.verbose),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wens/Projects/GNN/utils.py", line 81, in get_average_degree
    dataset = TUDataset(root='data/TUDataset',
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 129, in __init__
    super().__init__(root, transform, pre_transform, pre_filter,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/in_memory_dataset.py", line 81, in __init__
    super().__init__(root, transform, pre_transform, pre_filter, log,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 115, in __init__
    self._process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 260, in _process
    self.process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 203, in process
    self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 36, in read_tu_data
    node_label = read_file(folder, prefix, 'node_labels', torch.long)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 100, in read_file
    return read_txt_array(path, sep=',', dtype=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 33, in read_txt_array
    return parse_txt_array(src, sep, start, end, dtype, device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 19, in parse_txt_array
    return torch.tensor([[to_number(x) for x in line.split(sep)[start:end]]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 4 at dim 1 (got 2)
```

---
highschool_ct2
```sh
Processing...
Traceback (most recent call last):
  File "/home/wens/Projects/GNN/main.py", line 193, in <module>
    handle_option(args.function)
  File "/home/wens/Projects/GNN/main.py", line 174, in handle_option
    calcualte_parameters()
  File "/home/wens/Projects/GNN/main.py", line 87, in calcualte_parameters
    get_average_degree(dataset, args.verbose),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wens/Projects/GNN/utils.py", line 81, in get_average_degree
    dataset = TUDataset(root='data/TUDataset',
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 129, in __init__
    super().__init__(root, transform, pre_transform, pre_filter,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/in_memory_dataset.py", line 81, in __init__
    super().__init__(root, transform, pre_transform, pre_filter, log,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 115, in __init__
    self._process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 260, in _process
    self.process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 203, in process
    self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 36, in read_tu_data
    node_label = read_file(folder, prefix, 'node_labels', torch.long)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 100, in read_file
    return read_txt_array(path, sep=',', dtype=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 33, in read_txt_array
    return parse_txt_array(src, sep, start, end, dtype, device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 19, in parse_txt_array
    return torch.tensor([[to_number(x) for x in line.split(sep)[start:end]]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 2 at dim 1 (got 4)
```

---
infectious_ct1  
```sh
Traceback (most recent call last):
  File "/home/wens/Projects/GNN/main.py", line 193, in <module>
    handle_option(args.function)
  File "/home/wens/Projects/GNN/main.py", line 174, in handle_option
    calcualte_parameters()
  File "/home/wens/Projects/GNN/main.py", line 87, in calcualte_parameters
    get_average_degree(dataset, args.verbose),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wens/Projects/GNN/utils.py", line 81, in get_average_degree
    dataset = TUDataset(root='data/TUDataset',
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 129, in __init__
    super().__init__(root, transform, pre_transform, pre_filter,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/in_memory_dataset.py", line 81, in __init__
    super().__init__(root, transform, pre_transform, pre_filter, log,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 115, in __init__
    self._process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 260, in _process
    self.process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 203, in process
    self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 36, in read_tu_data
    node_label = read_file(folder, prefix, 'node_labels', torch.long)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 100, in read_file
    return read_txt_array(path, sep=',', dtype=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 33, in read_txt_array
    return parse_txt_array(src, sep, start, end, dtype, device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 19, in parse_txt_array
    return torch.tensor([[to_number(x) for x in line.split(sep)[start:end]]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 4 at dim 1 (got 2)
```

---
infectious_ct2
```sh
Processing...
Traceback (most recent call last):
  File "/home/wens/Projects/GNN/main.py", line 193, in <module>
    handle_option(args.function)
  File "/home/wens/Projects/GNN/main.py", line 174, in handle_option
    calcualte_parameters()
  File "/home/wens/Projects/GNN/main.py", line 87, in calcualte_parameters
    get_average_degree(dataset, args.verbose),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wens/Projects/GNN/utils.py", line 81, in get_average_degree
    dataset = TUDataset(root='data/TUDataset',
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 129, in __init__
    super().__init__(root, transform, pre_transform, pre_filter,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/in_memory_dataset.py", line 81, in __init__
    super().__init__(root, transform, pre_transform, pre_filter, log,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 115, in __init__
    self._process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 260, in _process
    self.process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 203, in process
    self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 36, in read_tu_data
    node_label = read_file(folder, prefix, 'node_labels', torch.long)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 100, in read_file
    return read_txt_array(path, sep=',', dtype=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 33, in read_txt_array
    return parse_txt_array(src, sep, start, end, dtype, device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 19, in parse_txt_array
    return torch.tensor([[to_number(x) for x in line.split(sep)[start:end]]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 4 at dim 1 (got 2)
```

---
mit_ct1 mit_ct2
```sh
Processing...
Traceback (most recent call last):
  File "/home/wens/Projects/GNN/main.py", line 193, in <module>
    handle_option(args.function)
  File "/home/wens/Projects/GNN/main.py", line 174, in handle_option
    calcualte_parameters()
  File "/home/wens/Projects/GNN/main.py", line 87, in calcualte_parameters
    get_average_degree(dataset, args.verbose),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wens/Projects/GNN/utils.py", line 81, in get_average_degree
    dataset = TUDataset(root='data/TUDataset',
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 129, in __init__
    super().__init__(root, transform, pre_transform, pre_filter,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/in_memory_dataset.py", line 81, in __init__
    super().__init__(root, transform, pre_transform, pre_filter, log,
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 115, in __init__
    self._process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/data/dataset.py", line 260, in _process
    self.process()
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/datasets/tu_dataset.py", line 203, in process
    self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 36, in read_tu_data
    node_label = read_file(folder, prefix, 'node_labels', torch.long)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/tu.py", line 100, in read_file
    return read_txt_array(path, sep=',', dtype=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 33, in read_txt_array
    return parse_txt_array(src, sep, start, end, dtype, device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/bt/lib/python3.12/site-packages/torch_geometric/io/txt_array.py", line 19, in parse_txt_array
    return torch.tensor([[to_number(x) for x in line.split(sep)[start:end]]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 4 at dim 1 (got 2)
```