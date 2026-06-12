from skbuild import setup


def _filter_wheel_manifest(files):
    excluded_prefixes = (
        "minion/include/",
        "minion/cec/",
        "cec_input_data/",
    )
    return [path for path in files if not path.startswith(excluded_prefixes)]


setup(
    include_package_data=False,
    packages=["minionpy"],
    cmake_process_manifest_hook=_filter_wheel_manifest,
)
