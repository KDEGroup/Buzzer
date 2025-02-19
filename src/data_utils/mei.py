import datasets
from pathlib import Path


def generate_mei(saved_root='cache/mei'):
    if not Path(saved_root).exists():
        Path(saved_root).mkdir(exist_ok=True, parents=True)
    
    cache_dir = Path(saved_root) / 'raw'
    data = datasets.load_dataset('ise-uiuc/Magicoder-Evol-Instruct-110K', 
                                cache_dir=cache_dir,
                                split='train')
    data = data.shuffle(42)

    mem_pretrain = data.select(range(0, 30000))
    non_shadow = data.select(range(30000, 50000))
    non_caliberate = data.select(range(50000, 70000))
    non_utils = data.select(range(70000, 90000))

    non_test = data.select(range(90000, 95000))
    mem_test = mem_pretrain.select(range(0, 5000))

    wb_mem_train = data.select(range(5000, 30000))
    wb_non_train = non_utils

    for each_data, name in zip(
        [mem_pretrain, non_shadow, non_caliberate, non_utils, non_test, mem_test, wb_mem_train, wb_non_train],
        ['mem_pretrain', 'non_shadow', 'non_calibrate', 'non_utils', 'non_test', 'mem_test', 'wb_mem_train', 'wb_non_train']
    ):
        save_dir = Path(saved_root) / f'{name}'
        each_data.save_to_disk(save_dir)


if __name__ == "__main__":
    generate_mei()
