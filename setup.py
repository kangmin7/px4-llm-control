from setuptools import setup
import os
from glob import glob

package_name = 'px4_llm_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kangmin',
    maintainer_email='kangmin7@gmail.com',
    description='Natural-language PX4 mission control via LLM-generated offboard steps',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission_executor = px4_llm_control.mission_executor:main',
            'command_cli = px4_llm_control.command_cli:main',
            'command_gui = px4_llm_control.command_gui:main',
        ],
    },
)
