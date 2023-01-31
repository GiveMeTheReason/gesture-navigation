import os

import yaml

CONFIG_FILENAME = os.getenv(
    'CONFIG_FILENAME') or os.path.join('config', 'config.yaml')


def _join_path(loader: yaml.Loader, node: yaml.Node) -> str:
    seq = loader.construct_sequence(node)
    return os.path.join(*map(str, seq))


def _zfill_id(loader: yaml.Loader, node: yaml.Node) -> str:
    value = loader.construct_yaml_int(node)
    return str(value).zfill(3)


def _add_tag_handlers(loader: yaml.Loader) -> None:
    loader.add_constructor('!join_path', _join_path)
    loader.add_constructor('!zfill_id', _zfill_id)


def _print_config(config: dict) -> None:
    import pprint
    pprint.pprint(config)


def get_config(filename: str = CONFIG_FILENAME, verbose: bool = False) -> dict:
    loader = yaml.SafeLoader
    _add_tag_handlers(loader)

    config = yaml.load(open(filename, 'r'), loader)
    if verbose:
        _print_config(config)
    return config


CONFIG = get_config(CONFIG_FILENAME, verbose=False)
