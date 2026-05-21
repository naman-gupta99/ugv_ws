from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ugv_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name,'launch'),glob(os.path.join('launch','*launch.py'))),
    ],
    install_requires=['setuptools', 'debugpy'],
    zip_safe=True,
    maintainer='dudu',
    maintainer_email='dudu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'keyboard_ctrl = ugv_tools.keyboard_ctrl:main',
            'joy_ctrl = ugv_tools.joy_ctrl:main',
            'behavior_ctrl = ugv_tools.behavior_ctrl:main',
            'pt_ctrl = ugv_tools.pt_ctrl:main',
            'llm_pt_ctrl = ugv_tools.llm_pt_ctrl:main',
            'align_ctrl = ugv_tools.align_ctrl:main',
            'distance_ctrl = ugv_tools.distance_ctrl:main',
            'inspection_pipeline = ugv_tools.inspection_pipeline:main',
            'run_inspection = ugv_tools.run_inspection:main',
            'ugv_model_smoke_test = ugv_tools.model_smoke_test:main',
            'wheel_encoder_diagnostic = ugv_tools.wheel_encoder_diagnostic:main',
        ],
    },
)
