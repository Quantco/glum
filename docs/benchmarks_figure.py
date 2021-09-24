import subprocess

base_cmd = (
    "glm_benchmarks_run --threads 6 --num_rows {n} --storage {s}"
    "--problem_name {p} --library_name {lib}"
)

problems = [
    "intermediate-insurance-no-weights-lasso-tweedie-p=1.5",
    "intermediate-insurance-no-weights-lasso-poisson",
    "intermediate-insurance-no-weights-lasso-gaussian",
    "intermediate-insurance-no-weights-lasso-gamma",
    "intermediate-insurance-no-weights-lasso-binomial",
    "intermediate-insurance-no-weights-l2-tweedie-p=1.5",
    "intermediate-insurance-no-weights-l2-poisson",
    "intermediate-insurance-no-weights-l2-gaussian",
    "intermediate-insurance-no-weights-l2-gamma",
    "intermediate-insurance-no-weights-l2-binomial",
]

libraries = ["quantcore-glm", "r-glmnet", "h2o"]

n = 500000

# run r-glmnet and h2o benchmarks, sparse storage works best.
s = "sparse"
for lib in ["r-glmnet", "h2o"]:
    for p in problems:
        cmd = base_cmd.format(n=n, s=s, p=p, lib=lib)
        print(cmd)
        subprocess.run(cmd.split(" "))

# run quantcore-glm benchmarks where auto storage works best.
lib = "quantcore-glm"
s = "auto"
for p in problems:
    cmd = base_cmd.format(n=n, s=s, p=p, lib=lib)
    print(cmd)
    subprocess.run(cmd.split(" "))

analyze_cmd = "glm_benchmarks_analyze --export"
subprocess.run(analyze_cmd)
