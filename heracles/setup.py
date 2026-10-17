from setuptools import find_packages, setup

setup(
    name="heracles",
    version="0.0.1",
    url="https://github.com/npolshakova/heracles",
    author="Aaron Ray, Nina Polshakova",
    author_email="aaronray@mit.edu, ninapolshakova@gmail.com",
    description="Cypher / Neo4j support for SparkDSG Scene graphs",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.yaml"]},
    install_requires=[
        "pyyaml",
        "neo4j-rust-ext",
        "numpy",
        "pytest",
        "openai",
        "parse",
        "spark-dsg",
        # Image-store reader (not on PyPI): pip install -e <Hydra>/hydra_image_store
        # or "hydra_image_store @ git+https://github.com/harelb/Hydra.git@feature/image-store#subdirectory=hydra_image_store"
        "hydra_image_store",
    ],
)
