import shutil
import pathlib
from typing import Dict, List, Optional


class Suite2PWrapperException(Exception):
    pass


def copy_and_add_uid(
        srcdir: pathlib.Path, dstdir: pathlib.Path,
        basenames: List[str], uid: Optional[str] = None) -> Dict[str, str]:
    """copy files matching basenames from a tree search of srcdir to
    dstdir with an optional unique id inserted into the basename.

    Parameters
    ----------
    srcdir : pathlib.Path
       source directory
    dstdir : pathlib.Path
       destination directory
    basenames : list
        list of basenames to copy
    uid : str
        uid to insert into basename (example a timestamp string)

    Returns
    -------
    copied_files : dict
        keys are basenames and vaues are output paths as strings

    """

    copied_files: dict = {}

    for basename in basenames:
        result = list(srcdir.rglob(basename))
        if len(result) == 0:
            # case for copying *.tif when non-existent
            continue
        if len(result) != 1:
            if "*" not in basename:
                raise ValueError(f"{len(result)} matches found in {srcdir} "
                                 f"for {basename}. Expected 1 match.")
        copied_files[basename] = []
        for iresult in result:
            dstbasename = iresult.name
            if uid is not None:
                dstbasename = iresult.stem + f"_{uid}" + iresult.suffix

            dstfile = dstdir / dstbasename

            shutil.copyfile(iresult, dstfile)

            copied_files[basename].append(str(dstfile))

    return copied_files