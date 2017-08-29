import os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def get_time(folder):
    return time.ctime(max(
        os.path.getmtime(root)
        for root, _, _ in os.walk(folder)))


def get_server(path, servers, default="this"):
    for server in servers:
        if server in path:
            return server
    return default


def find_first_digit(string):
    for i, letter in enumerate(string):
        if letter.isdigit():
            return i
    return len(string)


def split(string):
    i = find_first_digit(string)
    return string[:i], float(string[i:])


def parse_info(stats_path, servers, default="this"):
    folder_name = stats_path.split("data")[0][:-1].split("/")[-1]
    server = get_server(stats_path.split(folder_name)[0], servers, default)
    info_arr = folder_name.split("_")
    info_arr.remove("results")
    info = dict(
        server=server,
        problem=info_arr[0])
    for item in info_arr[1:]:
        key, val = split(item)
        info[key] = val
    return info


def func(t, t_0, A, tau):
    return A * (1.-np.exp(-(t-t_0)/tau))


os.chdir(os.getenv("HOME"))
thishost = os.uname()[1]

servers = ["lizard",
           "scissors",
           "ceylon",
           "los",
           "spock",
           "paper",
           "duke"]

basedirs = ["git/Oasis",
            "lizard/Oasis",
            "scissors/git/Oasis",
            "ceylon/Oasis",
            "los/Oasis",
            "lscr_spock/mads/git/Oasis",
            "paper/git/Oasis",
            "duke/git/Oasis",
            "duke/lizard/Oasis",
            "duke/scissors/git/Oasis",
            "duke/ceylon/Oasis",
            "duke/los/Oasis",
            "duke/lscr_spock/mads/git/Oasis",
            "duke/paper/git/Oasis"]

max_age = 1*60*60*1.  # 1 hour

t_cur = time.time()

sims = []
for basedir in basedirs:
    if bool(os.path.exists(basedir) and os.path.isdir(basedir)):
        for folder in os.listdir(basedir):
            path = os.path.join(basedir, folder)
            data_path = os.path.join(path, "data")
            if bool(os.path.isdir(path) and os.path.exists(data_path)):
                for sim in os.listdir(data_path):
                    sim_path = os.path.join(data_path, sim)
                    stats_path = os.path.join(sim_path, "Stats",
                                              "dump_flux.dat")
                    if bool(os.path.isdir(sim_path) and
                            os.path.exists(stats_path)):
                        t_mod = os.path.getmtime(stats_path)
                        age = t_cur-t_mod
                        if age < max_age:
                            sims.append((stats_path, t_mod))

for stats_path, t_mod in sims:
    data = np.loadtxt(stats_path)
    info = parse_info(stats_path, servers, default=thishost)
    # print info, data.shape
    t = data[:, 1]
    Re = data[:, 8]
    plt.plot(t, Re, 'o',
             label="{server}: N={N}, F={F}".format(**info))
    p0 = (0., 2000., 1000.)
    try:
        popt, pcov = curve_fit(func, t, Re, p0)
        plt.plot(t, func(t, *popt), '-',
                 label="fit, A={1:4.2f}, tau={2:4.2f}".format(*popt))
    except:
        pass

plt.xlabel("Time")
plt.ylabel("Re")
plt.legend()
plt.show()
