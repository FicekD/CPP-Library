import argparse
import os
import xml
import warnings

from dataclasses import dataclass
import xml.etree
import xml.etree.ElementTree

import numpy as np
import scipy


warnings.filterwarnings('error')


@dataclass
class GenerationDefinition:

    @dataclass
    class Resolution:
        width: int
        height: int

    @dataclass
    class Option:
        name: str
        weight: float
    
    @dataclass
    class RangeOption(Option):
        limits: tuple[float, float]

    n: int
    dtype: np.dtype

    dims: list[Resolution]

    generation_options: list[RangeOption]
    sign_options: list[Option]
    transform_options: list[Option]
    property_options: list[Option]


@dataclass
class TestDefinition:
    name: str
    unary: bool
    numpy_fn: callable


@dataclass
class TestResult:
    input_indices: list[int]
    output: np.ndarray

    def serialize(self) -> bytes:
        pass


def get_generation_definition(path: str) -> GenerationDefinition:
    tree = xml.etree.ElementTree.parse(path)
    root = tree.getroot()
    
    n = int(root.find('N').text)
    dtype = eval('np.' + ''.join(root.find('Data').find(k).text for k in ['Type', 'Bits']))

    dims = [GenerationDefinition.Resolution(int(r.find('Width').text), int(r.find('Height').text)) for r in root.find('Dims').findall('Option')]

    parse_options_fn = lambda _root: [
        GenerationDefinition.Option(
            o.find('Name').text.lower(),
            float(o.find('Weight').text))
        for o in _root.findall('Option')]
    
    parse_range_options_fn = lambda _root: [
        GenerationDefinition.RangeOption(
            o.find('Name').text.lower(),
            float(o.find('Weight').text),
            tuple(float(o.find(k).text) for k in ['Min', 'Max']))
        for o in _root.findall('Option')]

    generation_options = parse_range_options_fn(root.find('Generation').find('Values'))
    sign_options = parse_options_fn(root.find('Generation').find('Sign'))
    transform_options = parse_options_fn(root.find('Generation').find('Transform'))
    property_options = parse_options_fn(root.find('Generation').find('Property'))

    return GenerationDefinition(
        n,
        dtype,
        dims,
        generation_options,
        sign_options,
        transform_options,
        property_options
    )


def get_test_definitions(path: str) -> dict[str, list[TestDefinition]]:
    tree = xml.etree.ElementTree.parse(path)
    root = tree.getroot()

    parse_tests_fn = lambda _root: [
        TestDefinition(
            t.find('Name').text.lower(),
            t.find('Unary').text == 'True',
            eval(t.find('PythonCall').text),
        )
        for t in _root.findall('Test')
    ]
    test_definitions = {child.tag: parse_tests_fn(child) for child in root}
    return test_definitions


def generate_inputs(definition: GenerationDefinition) -> list[np.ndarray]:
    def roll_for_weights(options: list[GenerationDefinition.Option]) -> GenerationDefinition.Option:
        weights = [o.weight for o in options]
        roll = np.random.uniform(0, sum(weights))
        cum_sum = 0
        for o in options:
            cum_sum += o.weight
            if roll <= cum_sum:
                return o
        return None
    
    sign_map = {
        'default': lambda _x: _x,
        'positive': lambda _x: np.abs(_x),
        'negative': lambda _x: -np.abs(_x),
    }

    transform_map = {
        'none': lambda _x: _x,
        'zeros': lambda _x: _x * (np.random.random_sample(_x.shape) > 0.5).astype(_x.dtype),
    }

    property_map = {
        'default': lambda _x: _x,
        'positivesemidefinite': lambda _x: _x @ _x.T,
    }

    inputs = list()
    for i in range(definition.n):
        resolution = np.random.choice(definition.dims)

        value_range = roll_for_weights(definition.generation_options).limits
        sign = roll_for_weights(definition.sign_options).name
        transform = roll_for_weights(definition.transform_options).name
        prop = roll_for_weights(definition.property_options).name

        mat = (value_range[1] - value_range[0]) * np.random.random_sample((resolution.height, resolution.width)) + value_range[0]
        mat = sign_map[sign](mat)
        mat = transform_map[transform](mat)
        mat = property_map[prop](mat)

        mat = mat.astype(definition.dtype)
        inputs.append(mat)
    return inputs


def get_outputs(inputs: list[np.ndarray], test_definition: TestDefinition) -> list[TestResult]:
    outputs = list()
    
    mat_indices = np.arange(len(inputs))
    for i in mat_indices:
        mat = inputs[i]

        if test_definition.unary:
            input_indices = (i, )
            op_inputs = (mat, )
        else:
            j = np.random.choice(mat_indices)
            while mat.shape != inputs[j].shape:
                j = np.random.choice(mat_indices)
                
            input_indices = (i, j)
            op_inputs = (mat, inputs[j])
        
        try:
            output = test_definition.numpy_fn(*op_inputs)
            if len(output.shape) == 0:
                output = output.reshape(1, 1)
            elif len(output.shape) == 1:
                output = output.reshape(-1, 1)
            output = output.astype(op_inputs[0].dtype)
        except:
            output = None

        outputs.append(TestResult(input_indices, output))
    return outputs


def write_bytes(path: str, data: bytes) -> None:
    with open(path, 'wb') as file:
        file.write(data)


def serialize_inputs(inputs: list[np.ndarray]) -> bytes:
    data = bytearray()
    for mat in inputs:
        data.extend(int(mat.shape[1]).to_bytes(4, 'little'))
        data.extend(int(mat.shape[0]).to_bytes(4, 'little'))

        data.extend(mat.tobytes())
    return data


def serialize_outputs(outputs: list[TestResult]) -> bytes:
    data = bytearray()
    for output in outputs:
        if output.output is None:
            continue
        data.extend(len(output.input_indices).to_bytes(4, 'little'))
        for index in output.input_indices:
            data.extend(int(index).to_bytes(4, 'little'))
        data.extend(int(output.output.shape[1]).to_bytes(4, 'little'))
        data.extend(int(output.output.shape[0]).to_bytes(4, 'little'))

        data.extend(output.output.tobytes())
    return data


def main(target_directory: str, generation_definition_path: str, test_definitions_path: str) -> None:
    os.makedirs(target_directory, exist_ok=True)
    
    generation_definition = get_generation_definition(generation_definition_path)
    test_definitions = get_test_definitions(test_definitions_path)

    np.random.seed(42)

    inputs = generate_inputs(generation_definition)
    input_bytes = serialize_inputs(inputs)
    write_bytes(os.path.join(target_directory, 'inputs.bin'), input_bytes)
    
    for test_module_name, test_definitions in test_definitions.items():
        test_module_dir = os.path.join(target_directory, test_module_name)
        os.makedirs(test_module_dir, exist_ok=True)
        print(f'{test_module_name} Module:')
        for test_definition in test_definitions:
            output = get_outputs(inputs, test_definition)
            print(f'\t{test_definition.name}: {len([o for o in output if o.output is not None])} cases')

            output_bytes = serialize_outputs(output)
            write_bytes(os.path.join(test_module_dir, f'{test_definition.name}.bin'), output_bytes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_directory', type=str)
    parser.add_argument('generation_definition_path', type=str)
    parser.add_argument('test_definitions_path', type=str)

    args = parser.parse_args()
    main(args.target_directory, args.generation_definition_path, args.test_definitions_path)
