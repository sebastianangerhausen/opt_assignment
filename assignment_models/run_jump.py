import subprocess

def run_julia():

    # subprocess.call(
    #     f"julia assignment_model.jl", shell=True
    # )
    subprocess.call(
        f"julia assignment_model.jl", shell=True
    )

if __name__ == "__main__":
    run_julia()

