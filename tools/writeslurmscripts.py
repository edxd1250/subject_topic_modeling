

import subprocess
import os
from pathlib import Path

def generate_slurm_script1(job_name, job_dir, emb_file, clean_file, output_dir, nodes=2, time='2:00:00', cpus_per_task=120, mem=500):
    script = f"""#!/bin/sh --login
#SBATCH -A data-machine
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --time={time}
#SBATCH --mem={mem}GB
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH -e /mnt/home/ande2472/jobrequests/outputs/sjrjobs/{job_name}/%x_%j_error_file.sb.e
#SBATCH -o /mnt/home/ande2472/jobrequests/outputs/sjrjobs/{job_name}/%x_%j_out_file.sb.o

module purge
module load Conda
module load CUDA

conda activate /mnt/home/ande2472/miniconda3/envs/hdbscan5


cd /mnt/home/ande2472/github/pubmed_topics/topic_model_pipeline/gen_topics
echo "Starting job on `hostname` at `date`"

python 3.1_generate_topics.py {emb_file} {clean_file} -o {output_dir}

scontrol show job $SLURM_JOB_ID

cd {job_dir}

sbatch testtopics2.sb
"""
    return script



def generate_slurm_script2(job_name, job_dir, emb_file, clean_file, output_dir, nodes=2, time='1:00:00', cpus_per_task=120, mem=200):
    script = f"""#!/bin/sh --login

#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --time={time}
#SBATCH --mem={mem}GB
#SBATCH --gpus=v100:2
#SBATCH -e /mnt/home/ande2472/jobrequests/outputs/sjrjobs/{job_name}/%x_%j_error_file.sb.e
#SBATCH -o /mnt/home/ande2472/jobrequests/outputs/sjrjobs/{job_name}/%x_%j_out_file.sb.o

module purge
module load Conda
module load CUDA/12.0.0

conda activate /mnt/home/ande2472/miniconda3/envs/hdbscan5


cd /mnt/home/ande2472/github/pubmed_topics/topic_model_pipeline/gen_topics
echo "Starting job on `hostname` at `date`"

python 3.2_generate_topics.py -o {output_dir}

scontrol show job $SLURM_JOB_ID

cd {job_dir}

sbatch testtopics3.sb
"""
    return script

def generate_slurm_script3(job_name, job_dir, emb_file, clean_file, output_dir, nodes=2, time='1:00:00', cpus_per_task=120, mem=200):
    script = f"""#!/bin/sh --login

#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --time={time}
#SBATCH --mem={mem}GB
#SBATCH --gpus=v100:2
#SBATCH -e /mnt/home/ande2472/jobrequests/outputs/sjrjobs/{job_name}/%x_%j_error_file.sb.e
#SBATCH -o /mnt/home/ande2472/jobrequests/outputs/sjrjobs/{job_name}/%x_%j_out_file.sb.o

module purge
module load Conda
module load CUDA

conda activate /mnt/home/ande2472/miniconda3/envs/hdbscan5


cd /mnt/home/ande2472/github/pubmed_topics/topic_model_pipeline/gen_topics
echo "Starting job on `hostname` at `date`"

python 3.3_generate_topics.py {clean_file} -o {output_dir}

scontrol show job $SLURM_JOB_ID

cd {job_dir}

sbatch testtopics4.sb
"""
    return script

def generate_slurm_script4(job_name, job_dir, emb_file, clean_file, output_dir, nodes=2, time='2:00:00', cpus_per_task=120, mem=367):
    script = f"""#!/bin/sh --login

#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --time={time}
#SBATCH --mem={mem}GB
#SBATCH --gpus=v100:2
#SBATCH -e /mnt/home/ande2472/jobrequests/outputs/sjrjobs/{job_name}/%x_%j_error_file.sb.e
#SBATCH -o /mnt/home/ande2472/jobrequests/outputs/sjrjobs/{job_name}/%x_%j_out_file.sb.o

module purge
module load Conda
module load CUDA/12.3.0

conda activate /mnt/home/ande2472/miniconda3/envs/hdbscan5


cd /mnt/home/ande2472/github/pubmed_topics/topic_model_pipeline/gen_topics
echo "Starting job on `hostname` at `date`"

python 3.4_generate_topics.py {emb_file} {clean_file} -o {output_dir}

scontrol show job $SLURM_JOB_ID

cd {job_dir}


"""
    return script

def main():
    job_name = input("Enter job name: ")
    generate_job = input("Custom job location? (yes/no): ").lower()
    if generate_job == 'yes':
        job_dir = input("Enter job directory location: ")
    else:
        job_dir = f'/mnt/home/ande2472/jobrequests/sjrjobs/{job_name}'
        print(f"Job dir at: {job_dir}")
    
    generate_out = input("Custom out location? (yes/no): ").lower()
    if generate_out == 'yes':
        output_dir = input("Enter output directory location: ")
    else:
        output_dir = f'/mnt/scratch/ande2472/sjrouts/{job_name}'
        print(f"Out dir at: {output_dir}")

    
    emb_file = input("Enter embedding location: ")
    clean_file = input("Enter clean docs location: ")

 
    custom = input("Do you want to customise resource requests? (yes/no): ").lower()
    if custom == "yes":
        print("For script 1:")
        nodes = int(input("Enter number of nodes: "))
        cpus_per_node = int(input("Enter number of cpus per node: "))
        mem = int(input("Enter memory (GB): "))
        time = input("Enter estimated time (format: days-hours:minutes:seconds): ")
        script1 = generate_slurm_script1(job_name, job_dir, emb_file, clean_file, output_dir, nodes, time, cpus_per_node, mem)

        print("For script 2:")
        nodes = int(input("Enter number of nodes: "))
        cpus_per_node = int(input("Enter number of cpus per node: "))
        mem = int(input("Enter memory (GB): "))
        time = input("Enter estimated time (format: days-hours:minutes:seconds): ")
        script2 = generate_slurm_script2(job_name, job_dir, emb_file, clean_file, output_dir, nodes, time, cpus_per_node, mem)

        print("For script 3:")
        nodes = int(input("Enter number of nodes: "))
        cpus_per_node = int(input("Enter number of cpus per node: "))
        mem = int(input("Enter memory (GB): "))
        time = input("Enter estimated time (format: days-hours:minutes:seconds): ")
        script3 = generate_slurm_script3(job_name, job_dir, emb_file, clean_file, output_dir, nodes, time, cpus_per_node, mem)

        print("For script 4:")
        nodes = int(input("Enter number of nodes: "))
        cpus_per_node = int(input("Enter number of cpus per node: "))
        mem = int(input("Enter memory (GB): "))
        time = input("Enter estimated time (format: days-hours:minutes:seconds): ")
        script4 = generate_slurm_script4(job_name, job_dir, emb_file, clean_file, output_dir, nodes, time, cpus_per_node, mem)


        
    else:

        script1 = generate_slurm_script1(job_name, job_dir, emb_file, clean_file, output_dir)
        script2 = generate_slurm_script2(job_name, job_dir, emb_file, clean_file, output_dir)
        script3 = generate_slurm_script3(job_name, job_dir, emb_file, clean_file, output_dir)
        script4 = generate_slurm_script4(job_name, job_dir, emb_file, clean_file, output_dir)

    job_dir = Path(job_dir)
    job_dir.mkdir(parents = True, exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents= True, exist_ok=True)
    script_file1 = job_dir/"testtopics1.sb"
    script_file2 = job_dir/f"testtopics2.sb"
    script_file3 = job_dir/f"testtopics3.sb"
    script_file4 = job_dir/f"testtopics4.sb"


    with open(script_file1, "w") as f:
        f.write(script1)

    print("SLURM script 1 generated successfully!")
    print(f"Script saved at: {script_file1}")

    with open(script_file2, "w") as f:
        f.write(script2)
    print("SLURM script 2 generated successfully!")
    print(f"Script saved at: {script_file2}")

    with open(script_file3, "w") as f:
        f.write(script3)
    print("SLURM script 3 generated successfully!")
    print(f"Script saved at: {script_file3}")

    with open(script_file4, "w") as f:
        f.write(script4)
    
    print("SLURM script 4 generated successfully!")
    print(f"Script saved at: {script_file4}")

    submit = input("Do you want to submit the job now? (yes/no): ").lower()
    if submit == "yes":
        os.chdir(job_dir)
        subprocess.run(["sbatch", "testtopics1.sb"])
        print("Job submitted successfully!")
    else:
        print("Job not submitted.")

if __name__ == "__main__":
    main()