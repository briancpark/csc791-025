# Final Course Project: Accelerating Video Super Resolution for Mobile Device

Running Super Resolution for video is a difficult task to do in real-time, and it's even more challenging for a resource constrained device such as a smartphone. You not only have to consider the compute and memory constraints of a mobile device, but also the battery life.

In this project, we explore and apply state of the art compression and optimization techniques to compress our model and accelerate it for mobile devices.

## Running the Project

To simply run everything, just run this command:

```sh
python3 final.py
```

If you want to run on bridges-2, then just submit a batch job:

```sh
sbatch job-gpu
```

There are a lot of comamnds associated with this project to learn how to use them do

```sh
python3 final.py -h
```
