from setuptools import setup, find_packages

setup(
    name='face_recognition_api',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'Flask',
        'opencv-python',
        'numpy',
        'face-recognition',
        # Add other dependencies as needed
    ],
    setup_requires=['wheel'],  # Add this line to resolve the 'bdist_wheel' issue
    entry_points={.\venv\Scripts\activate

        'console_scripts': [
            'your_script_name = PythonApplication1:main_function',
        ],
    },
)
