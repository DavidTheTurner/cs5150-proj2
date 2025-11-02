
# Prerequisites
1. Python installed
2. Latex installed w/ `latexmk`

# Setup

Run the following command
```terminal
pip install -r requirements.txt
```

# To Run
For Project 3, simply run the `run_project3` shell script

```terminal
./run_proj3
```

This script will run the code in `project3`, which has been provided by Gallier and updated to include our
implementations of the desired algorithms. It then builds the latex file `./CurveInterpolation/curve_interpolation.tex`
and saves the resulting PDF in the `output` directory that is generated when we run `project3` under
`output/report/report.pdf`. The script then packages the results into an output.zip, which can then be turned in.
