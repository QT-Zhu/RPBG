import numpy as np

# designed for actual application, set a random set for evaluation
def split_by_ratio(lists, train_ratio):
    
    sz = [len(l) for l in lists]
    assert len(set(sz)) == 1, f'list sizes differ {sz}'
    
    splits = []
    train_inds, val_inds = [], []

    train_n = int(sz[0] * train_ratio)
    train_inds, val_inds = np.split(np.random.permutation(sz[0]), [train_n])

    for lst in lists:
        lst = np.array(lst)
        #splits.append([lst[train_inds], lst[train_inds]])
        splits.append([lst[train_inds], lst[val_inds]])
    return splits

# for fixed split
def split_by_list(lists, train_list, eval_list):
    sz = [len(l) for l in lists]
    assert len(set(sz)) == 1, f'list sizes differ {sz}'
    with open(train_list, "r") as f:
        train_list = [line.strip() for line in f.readlines()]
    # print(train_list)
    with open(eval_list, "r") as f:
        eval_list = [line.strip() for line in f.readlines()]
    # print(eval_list)
    
    splits = []
    train_inds, val_inds = [], []

    train_inds = [index for index, image in enumerate(lists[1]) if image.split("/")[-1] in train_list]
    val_inds = [index for index, image in enumerate(lists[1]) if image.split("/")[-1] in eval_list]

    
    for lst in lists:
        lst = np.array(lst)
        splits.append([lst[train_inds], lst[val_inds]])
        
    return splits

# for datasets with no validation set
def split_identical(lists, shuffle=False):
    
    sz = [len(l) for l in lists]
    assert len(set(sz)) == 1, f'list sizes differ {sz}'
    
    splits = []
    train_inds = range(sz[0])

    for lst in lists:
        lst = np.array(lst)
        splits.append([lst[train_inds], lst[train_inds]])
    return splits