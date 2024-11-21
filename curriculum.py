import subprocess
import sys
import torch
import yaml
import numpy as np
from enum import Enum
from pathlib import Path
from omg.si.single_stochastic_interpolant_identity import SingleStochasticInterpolantIdentity

class ValidLessons(Enum):
    species = 0
    position = 1
    lattice = 2

def write_lesson(filename:str, lessons:ValidLessons):
    '''
    Write a config file for a particular "lesson" in curriculum learning style
    whereby only one interpolant will be trained

    :param filename:
        Name of file to alter
    :type filename: str
    :param lesson:
        Which lesson
    :type lesson: ValidLessons
    '''

    # Load yaml
    with open(filename, 'r') as file:
        template = yaml.safe_load(file)
        file.close()

    # Get interpolant and swap with identity
    mask_path = Path('omg.si.single_stochastic_interpolant_identity.SingleStochasticInterpolantIdentity')
    interpolants = template['model']['si']['init_args']['stochastic_interpolants']
    costs = template['model']['relative_si_costs'][i]
    for i in range(len(interpolants)):
        if i not in lessons:
            costs = (len(lessons) + 1) / len(costs)
            interpolants[i] = mask_path
        else:
            template['model']['relative_si_costs'][i] = 0.0

    # Save lesson
    base, ext = filename.split(".", 1)
    lesson_slice = ''
    for lesson in lessons:
        lesson_slice += f'{lesson}_'
    lesson_name = f"{base}_{lesson_slice}.{ext}"
    with open(f'lessons/{lesson_name}') as file:
        yaml.safe_dump(template, lesson_name)

if __name__ == '__main__':

    # Get config file
    filename = sys.argv[1]

    # Write a lesson
    lesson = ValidLessons(1)
    write_lesson(filename, lesson)
