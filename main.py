#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

has_data = True

is_classification = True

get_input = True

DIM = 2

max_data = 1200
max_train_data = 1200

train_points = []
test_points = []
group = 2
bound = (0, 1)
m_bound = (0.05, 0.5)

MIN_VALUE = 0
MAX_VALUE = 1

# global clusters

def read(name):
    # Read data
    with open(name) as csv_file:
        classes = []
        has_header = False
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        result = []

        for row in csv_reader:
            # if line_count > max_data:
            #     break

            # if (has_header and line_count == 0) or (not has_header and line_count == 1):
            #     DIM = len(row) - 1

            line_count += 1

            if has_header and line_count == 1:
                continue

            if is_classification:
                if row[-1] in classes:
                    row[-1] = classes.index(row[-1])
                else:
                    classes.append(row[-1])
                    row[-1] = len(classes) - 1

            row = np.array(row).astype(np.float)

            # if max_train_data > line_count:
            result.append(row)
            # else:
            #     test_points.append(row)

            # min = np.min(row[:-1])
            # max = np.max(row[:-1])
            # if MIN_VALUE > min:
            #     MIN_VALUE = min
            # if MAX_VALUE < max:
            #     MAX_VALUE = max

        # if is_classification:
        #     # group = len(classes)
        #     group = 7
        # else:
        #     group = 10
        # test_points = np.array(random.choices(train_points, k=600))
        # train_points = np.array(random.choices(train_points, k=1200))
        # clusters = len(classes)
        print('\033[93m' + str(len(classes)) + '\033[0m')
        return result


if has_data:

    group = 7
    if is_classification:
        if get_input:
        #     train_points = np.array(read('5clstrain1500.csv'))
        # test_points = np.array(read('5clstest5000.csv'))
            train_points = np.array(read('2clstrain1500.csv'))
        test_points = np.array(read('2clstest5000.csv'))
    else:
        train_idx = np.random.randint(0, 2000, 1200)
        train_idx.sort()
        orig = np.array(read('regdata2000 (2).csv'))
        if get_input:
            train_points = orig[train_idx]
            # train_points = np.array(read('4clstrain1200.csv'))
        # test_points = train_points.copy()
        test_points = orig.copy()
        # test_points = np.array(read('4clstest4000.csv'))
    # test_points = np.array(random.choices(train_points, k=600))
    # train_points = np.array(random.choices(train_points, k=1200))

    DIM = len(test_points[0]) - 1

    # Preprocess
    if is_classification:
        if get_input:
            train_points[:, :-1] /= train_points.max()
        test_points[:, :-1] /= test_points.max()
    else:
        if get_input:
            train_points /= train_points.max()
        test_points /= test_points.max()
    MAX_VALUE = 1

MIN_STRATEGY = (MAX_VALUE - MIN_VALUE) / 100
MAX_STRATEGY = (MAX_VALUE - MIN_VALUE) / 10

ratio = 1
# ratio = M / bound[1]
train_points_of_group = 300
test_points_of_group = 50

colors = [[np.random.ranf() for _ in range(3)] for _ in range(group)]


def app(a, b):
    a = a.copy()
    for element in b:
        a.append(element)
    return a


# for i in range(group):
#     center = [np.random.ranf() * bound[1] for _ in range(DIM)]
#     m = np.random.ranf() * (m_bound[1] - m_bound[0]) + m_bound[0]
#     tr_points = np.random.normal(size=[train_points_of_group, DIM], loc=center, scale=m)
#     te_points = np.random.normal(size=[test_points_of_group, DIM], loc=center, scale=m)
#     train_group_points = np.zeros((train_points_of_group, DIM + 1))
#     test_group_points = np.zeros((test_points_of_group, DIM + 1))
#     train_group_points[:, :-1] = tr_points
#     train_group_points[:, -1] = np.full(train_points_of_group, i)
#     test_group_points[:, :-1] = te_points
#     test_group_points[:, -1] = np.full(test_points_of_group, i)
#     train_points = app(train_points, train_group_points)
#     test_points = app(test_points, test_group_points)
#     plt.scatter(train_group_points[:, 0], train_group_points[:, 1], color=colors[i])
#     # plt.pause(0.05)
# plt.show()

IND_SIZE = (DIM + 1) * group


# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    a = [random.uniform(imin, imax) for _ in range(size)]
    ind = icls(a)
    b = [random.uniform(smin, smax) for _ in range(size)]
    ind.strategy = scls(b)
    return ind


def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children

        return wrappper

    return decorator


def get_weights(G, y):
    try:
        if is_classification:
            b = np.zeros((len(y), group))
            b[np.arange(len(y)), y.astype(int)] = 1
        else:
            b = y
        trans = np.transpose(G)
        inv = np.linalg.pinv(np.matmul(trans, G))
        return np.matmul(np.matmul(inv, trans), b)
    except Exception as exp:
        if is_classification:
            return np.random.randint(0, 2, size=[group, len(y)])
        else:
            return np.random.randint(0, 2, size=group)
        # return np.ones([group, len(y)])


def accuracy(ind, data):
    vis = [pp for pp in ind]
    V = np.array(vis[0:DIM * group]).reshape([group, DIM])
    G = []
    label = np.array(data)[:, -1]
    for point in data:
        G_row = []
        for g in range(group):
            X = np.array(point)
            tr = np.transpose(X[: -1] - V[g])
            pow = ((-ind[DIM * group + g] * ratio) * np.matmul(tr, (X[: -1] - V[g])))
            v = np.math.e ** pow
            G_row.append(v)
        G.append(G_row)

    W = get_weights(G, label)
    Y = np.matmul(G, W)
    true = 0
    res = []
    for el in range(len(Y)):
        if label[el] == np.argmax(Y[el]):
            true += 1
            res.append(True)
        else:
            res.append(False)
    return true / len(label), res


def error(ind, data):
    vis = [pp for pp in ind]
    # print(vis)
    V = np.array(vis[0:DIM * group]).reshape([group, DIM])
    G = []
    label = np.array(data)[:, -1]
    for point in data:
        G_row = []
        for g in range(group):
            X = np.array(point)
            tr = np.transpose(X[:-1] - V[g])
            pow = ((-ind[DIM * group + g] * ratio) * np.matmul(tr, (X[: -1] - V[g])))
            v = np.math.e ** pow
            G_row.append(v)
        G.append(G_row)

    W = get_weights(G, label)
    Y = np.matmul(G, W)
    r_res = []

    if is_classification:
        for i in range(len(Y)):
            r_res.append(np.argmax(Y[i]))

        np_res = np.array(r_res)
    else:
        np_res = Y

    trY = np.transpose(np_res - label)
    res = 0.5 * np.matmul(trY, (np_res - label))
    return res


def res_reg(ind, data):
    vis = [pp for pp in ind]
    # print(vis)
    V = np.array(vis[0:DIM * group]).reshape([group, DIM])
    G = []
    for point in data:
        G_row = []
        for g in range(group):
            X = np.array(point)
            tr = np.transpose(X[:-1] - V[g])
            pow = ((-ind[DIM * group + g] * ratio) * np.matmul(tr, (X[: -1] - V[g])))
            v = np.math.e ** pow
            G_row.append(v)
        G.append(G_row)
    Y1 = np.array(data)[:, -1]
    W = get_weights(G, Y1)
    Y = np.matmul(G, W)

    return Y


def e(ind):
    return error(ind, train_points),


toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                 IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", e)
toolbox.register("evaluate", benchmarks.sphere)

toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))


def show_plot(res, data, ind):
    centers = np.array(ind[0:DIM * group]).reshape([group, DIM])
    radius = ind[DIM * group:]
    r = np.full(len(data), 'red')
    g = np.full(len(data), 'green')
    plt.clf()
    ax = plt.gca()
    ax.cla()
    colors = np.where(res, g, r)
    plt.scatter(data[:, 0], data[:, 1], color=colors)
    plt.scatter(centers[:, 0], centers[:, 1], color='black')
    for cir in range(len(radius)):
        circle = plt.Circle(centers[cir], radius[cir], color='black', fill=False)
        # circle = plt.(centers[cir], radius[cir], color='black', fill=False)
        ax.add_artist(circle)
    plt.show()


def show_plot_reg(res, data):
    plt.clf()
    plt.scatter(np.arange(len(data)), data[:, -1], color='red', s=3.5)
    plt.scatter(np.arange(len(res)), res, color='black', s=3)
    plt.show()


def write_ind(ind):
    with open('ind', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        chrom = [pop for pop in ind]

        employee_writer.writerow(chrom)


def read_ind():
    with open('ind') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            return np.array(row).astype(np.float)


def main():
    if get_input:
        plt.clf()
        plt.xlabel('Iteration')
        plt.ylabel('Error')

        random.seed()
        MU, LAMBDA = 10, 70
        pop = toolbox.population(n=MU)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(e)
        # stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        b_num = 5

        up = 0
        checkpoint_score = 100000
        best_pop = None
        iteration = 0
        score_list = []

        while True:
            score = 0
            pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                                      cxpb=0.6, mutpb=0.3, ngen=b_num, stats=stats, halloffame=hof)
            for data in logbook:
                score += data['std']
            score /= b_num

            if score < checkpoint_score:
                checkpoint_score = score
                up = 0
                best_pop = pop
                print('\033[93m' + 'Checkpoint set' + '\033[0m')
            else:
                up += 1

            if (up * b_num > 20 and iteration * b_num > 40) or iteration * b_num > 300:
                # if iteration * b_num >= 0:
                break

            score_list.append(score)

            iteration += 1

            # plt.plot(range(iteration), score_list, '-o', color='black')
            # plt.pause(0.01)

        print("Finished!")
        print(f'iterations: {iteration}')
        # print(f'clusters: {clusters}')


        # plt.show()

        presenter = None
        max_fit = 0
        for ind in best_pop:
            if ind.fitness.values[0] > max_fit:
                presenter = ind
                max_fit = ind.fitness.values[0]

        write_ind(presenter)

    else:
        presenter = read_ind()

    if get_input:
        if is_classification:
            tr_acc = accuracy(presenter, train_points)
            print("Train Acc: ", tr_acc[0])
        print("Train Err: ", error(presenter, train_points))
        if is_classification:
            show_plot(tr_acc[1], train_points, presenter)
        else:
            rr = res_reg(presenter, train_points)
            show_plot_reg(rr, train_points)

    if is_classification:
        te_acc = accuracy(presenter, test_points)
        print("Test Acc: ", te_acc[0])
    print("Test Err: ", error(presenter, test_points))
    if is_classification:
        show_plot(te_acc[1], test_points, presenter)
    else:
        rr = res_reg(presenter, test_points)
        show_plot_reg(rr, test_points)

    return pop, logbook, hof


if __name__ == "__main__":
    main()
