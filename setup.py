from setuptools import setup, find_packages

with open('readme.md', 'r', encoding='utf-8') as f:
    long_description = f.read()
    
with open('requirements.txt', 'r', encoding='utf-8') as pr:
    install_requires = pr.readlines()


setup(
    name="lexsiai",
    use_scm_version=True,                  # version comes from git tags
    setup_requires=["setuptools_scm"],     # required at build time
    description="Full stack ML Observability with Lexsi.ai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="lexsi, lexsiai, ML observability, observability, machine learning",
    license="Lexsi Labs Source Available License (LSAL) v1.0",
    url="https://docs.lexsi.ai",
    project_urls={
        "Homepage": "https://docs.lexsi.ai",
        "Repository": "https://github.com/Lexsi-Labs/lexsiai-sdk",
        "Issues": "https://github.com/Lexsi-Labs/lexsiai-sdk/issues",
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=install_requires,
    include_package_data=True,   # together with MANIFEST.in
    package_data={
        "": ["*.md", "*.txt"],
        # example: include env templates if you have them inside the package
        # "lexsiai": ["common/config/.env.*"],
    },
)
