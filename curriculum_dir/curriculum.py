import subprocess
from argparse import ArgumentParser
import torch
import yaml
import numpy as np
from enum import Enum
from pathlib import Path
from omg.si.single_stochastic_interpolant_identity import SingleStochasticInterpolantIdentity

class ValidLessons(Enum):
    species = 0
    pos = 1
    cell = 2

def write_lesson(filename:str, lessons:list):
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
    mask_path = 'omg.si.single_stochastic_interpolant_identity.SingleStochasticInterpolantIdentity'
    interpolants = template['model']['si']['init_args']['stochastic_interpolants']
    distributions = template['model']['sampler']['init_args']
    costs = template['model']['relative_si_costs']
    valid_lessons = list(ValidLessons)
    lesson_ints = [element.value for element in lessons]
    for i in range(len(interpolants)):
        if i in lesson_ints:
            interpolants[i] = interpolants[i]
            costs[i] = 1 / (len(lessons))
        else:
            distributions[f'{valid_lessons[i].name}_distribution'] = {'class_path' : 'omg.sampler.distributions.MirrorData'}
            interpolants[i]['class_path'] = mask_path
            interpolants[i]['init_args'] = {}
            costs[i] = 0.0

    # Save lesson
    template['model']['si']['init_args']['stochastic_interpolants'] = interpolants
    template['model']['relative_si_costs'] = costs
    base, ext = filename.split(".", 1)
    lesson_slice = ''
    for lesson in lessons:
        lesson_slice += f'_{lesson.name}'
    lesson_name = f"{base}{lesson_slice}.{ext}"
    with open(f'lessons/{lesson_name}', 'w') as file:
        yaml.safe_dump(template, file)

if __name__ == '__main__':

    # Get config file
    parser = ArgumentParser(prog="FlowMMConverter", description="convert FlowMM data to xyz files")
    parser.add_argument("--lessons", type=str, nargs='+', help="Lessons to train model")
    parser.add_argument("--template", type=str, help="Path to template file")
    args = parser.parse_args()

    # Write a lesson
    lesson = [ValidLessons[element] for element in args.lessons]
    write_lesson(args.template, lesson)
