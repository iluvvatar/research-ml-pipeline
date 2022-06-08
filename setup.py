from setuptools import setup, find_packages


def get_requirements():
    packages = []
    with open('requirements.txt') as f:
        for line in f:
            line = line.strip()
            if line:
                packages.append(line)
    return packages


setup(
    name='mlpipeline',
    version='0.0.1',
    description='Research ML pipeline',
    url='https://github.com/iluvvatar/research-ml-pipeline',
    author='Malakhov Ilya',
    author_email='malakhov.ilya.pavlovich@yandex.ru',
    license='MIT',
    packages=find_packages(where='.',),
    install_requires=get_requirements(),
)
