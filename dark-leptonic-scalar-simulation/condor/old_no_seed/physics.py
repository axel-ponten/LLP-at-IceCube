import argparse

parser = argparse.ArgumentParser()
parser.add_argument("jobnum", type = int)
args = parser.parse_args()

jobnum = args.jobnum

outfile = open("/data/user/axelpo/test_condor_job_{}.txt".format(jobnum), "w")
outfile.write("using icetray-axel " + str(jobnum))
outfile.close()
