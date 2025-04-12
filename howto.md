1. Compile and istall locally (in a virtual environment) `python -m pip install -e .`
2. Test!
3. Compile the project `python -m build`
4. Rename the linux wheel in **dist/** to **symjit-1.4.1-cp312-cp312-manylinux_2_34_x86_64.whl**
5. Add the wheel file to git
6. Commit and pull to github (**https://github.com/siravan/symjit**)
7. On both windows and darwin, pull, build, test, add wheel, commit, and upload.
8. Pull everything to linux.
9. Upload to PyPi as `python -m twine upload dist/*`
10. Goto **symjit-feedstock** directory
11. Edit recipe/meta.yaml by changing version and SHA256 (copy from PyPi)
12. Commit and push to github (**https://github.com/shahriariravanian/symjit-feedstock**).
13. On Github, submit a pull request to **https://github.com/conda-forge/symjit-feedstock**
14. Done!
