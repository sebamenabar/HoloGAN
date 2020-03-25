import os
import errno
from itertools import chain, starmap


def transform_curriculum(epoch):
    if epoch < 10:
        return {
            "azimuth_range": (0, 0),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (1.0, 1.0),
            "tx_range": (0, 0),
            "ty_range": (0, 0),
            "tz_range": (0, 0),
        }
    elif (epoch >= 10) and (epoch < 15):
        return {
            "azimuth_range": (-5, 5),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (0.9, 1.0),
            "tx_range": (-0.5, 0.5),
            "ty_range": (-0.5, 0.5),
            "tz_range": (-0.5, 0.5),
        }
    elif (epoch >= 15) and (epoch < 20):
        return {
            "azimuth_range": (-10, 10),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (0.8, 1.0),
            "tx_range": (-1.0, 1.0),
            "ty_range": (-1.0, 1.0),
            "tz_range": (-1.0, 1.0),
        }
    elif (epoch >= 20) and (epoch < 25):
        return {
            "azimuth_range": (-15, 15),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (0.7, 1.0),
            "tx_range": (-1.5, 1.5),
            "ty_range": (-1.5, 1.5),
            "tz_range": (-1.5, 1.5),
        }
    elif (epoch >= 25) and (epoch < 30):
        return {
            "azimuth_range": (-25, 25),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (0.6, 1.0),
            "tx_range": (-2, 2),
            "ty_range": (-2, 2),
            "tz_range": (-2, 2),
        }
    elif (epoch >= 30) and (epoch < 35):
        return {
            "azimuth_range": (-40, 40),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (0.5, 1.0),
            "tx_range": (-3.5, 3.5),
            "ty_range": (-3.5, 3.5),
            "tz_range": (-3.5, 3.5),
        }
    elif (epoch >= 35) and (epoch < 40):
        return {
            "azimuth_range": (-60, 60),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (0.4, 1.0),
            "tx_range": (-4, 4),
            "ty_range": (-4, 4),
            "tz_range": (-4, 4),
        }
    elif (epoch >= 40) and (epoch < 45):
        return {
            "azimuth_range": (-80, 80),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (0.4, 1.1),
            "tx_range": (-4.5, 4.5),
            "ty_range": (-4.5, 4.5),
            "tz_range": (-4.5, 4.5),
        }
    elif (epoch >= 45) and (epoch < 50):
        return {
            "azimuth_range": (-110, 110),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (0.4, 1.2),
            "tx_range": (-5, 5),
            "ty_range": (-5, 5),
            "tz_range": (-5, 5),
        }
    elif (epoch >= 50) and (epoch < 55):
        return {
            "azimuth_range": (-140, 140),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (0.4, 1.3),
            "tx_range": (-5.5, 5.5),
            "ty_range": (-5.5, 5.5),
            "tz_range": (-5.5, 5.5),
        }
    elif epoch >= 55:
        return {
            "azimuth_range": (-180, 180),
            "elevation_range": (0, 0),
            "roll_range": (0.0, 0.0),
            "scale_range": (0.4, 1.3),
            "tx_range": (-5.5, 5.5),
            "ty_range": (-5.5, 5.5),
            "tz_range": (-5.5, 5.5),
        }


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def cfg_to_exp_name(cfg):
    return


def flatten_json_iterative_solution(dictionary):
    """Flatten a nested json file"""

    def unpack(parent_key, parent_value):
        """Unpack one level of nesting in json file"""
        # Unpack one level only!!!

        if isinstance(parent_value, dict):
            for key, value in parent_value.items():
                temp1 = parent_key + "." + key
                yield temp1, value
        elif isinstance(parent_value, list):
            i = 0
            for value in parent_value:
                temp2 = parent_key + "." + str(i)
                i += 1
                yield temp2, value
        else:
            yield parent_key, parent_value

    # Keep iterating until the termination condition is satisfied
    while True:
        # Keep unpacking the json file until all values are atomic elements (not dictionary or list)
        dictionary = dict(chain.from_iterable(starmap(unpack, dictionary.items())))
        # Terminate condition: not any value in the json file is dictionary or list
        if not any(
            isinstance(value, dict) for value in dictionary.values()
        ) and not any(isinstance(value, list) for value in dictionary.values()):
            break

    return dictionary
