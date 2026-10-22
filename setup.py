from setuptools import setup, find_packages

setup(
    name='IzzyViz',
    version='0.1.0',
    description='A library for visualizing attention scores in transformer models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Luo Xizi',
    author_email='e0909010@u.nus.edu',
    url='https://github.com/WING-NUS/IzzyViz',  
    packages=find_packages(),
    # Runtime dependencies: only what izzyviz/ actually imports.
    # transformers and pybind11 were previously listed here but are never
    # imported by the library -- they are needed only to run the examples,
    # so they now live in the 'examples' extra.
    install_requires=[
        'matplotlib>=3.5.0',
        # The <2.0.0 upper bound is retained deliberately, not by inertia.
        # The library imports and renders correctly under numpy 2.2.6, but
        # check_stability_heatmap_with_gradient_color calls np.array() on a
        # list of torch tensors and emits a numpy-2 DeprecationWarning ("__array__ implementation doesn't
        # accept a copy keyword"), which becomes an error in a future numpy.
        # Fix that call site first, then relax this bound with test evidence
        # across both numpy majors. See the maintenance report, finding R5.
        'numpy>=1.21.0,<2.0.0',
        'torch>=1.10.0',
        'pandas>=1.4.0',
    ],
    extras_require={
        # Needed to reproduce the README examples, which load real models.
        # >=4.36 for from_pretrained(attn_implementation="eager"), which the
        # examples need: SDPA, the default since transformers 5, returns no
        # attention weights.
        'examples': ['transformers>=4.36'],
        'dev': ['pytest>=7.0', 'ruff>=0.4'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.9',
    license='MIT', 
)
