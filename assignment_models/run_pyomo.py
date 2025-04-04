import pyomo.environ as pyo
import logging
from datetime import datetime
import numpy as np
import tracemalloc

from data_reader import load_data,load_result

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

def assignment_model(lp_solve=False):

    model = pyo.AbstractModel()

    model.D = pyo.Set()
    model.H = pyo.Set()
    model.C = pyo.Set()
    model.S = pyo.Set()
    model.DHS = pyo.Set()
    model.SC = pyo.Set()
    model.DHSC = pyo.Set()
    model.mixedC = pyo.Set()
    model.mixedH = pyo.Set()
    model.FulltimeS = pyo.Set()
    model.valid_DS = pyo.Set()
    model.valid_S = pyo.Set()

    model.centerDemand = pyo.Param(model.D, model.H, model.C, default=0,mutable=True)
    model.specialistHourlyCapacity = pyo.Param(model.S, default=0)
    model.specialistPreference = pyo.Param(model.SC, default=0)
    model.centerMaxBacklogSize = pyo.Param(model.D, model.H, model.C, default=0)
    model.centerFullTimeCoverPerc = pyo.Param(model.mixedC, default=0)

    model.specialistAtCenterByHour = pyo.Var(model.DHSC, domain=pyo.Binary, )
    model.specialistWorkloads = pyo.Var(model.DHSC, domain=pyo.NonNegativeReals, )
    model.centerHourlyUnmetDemand = pyo.Var(model.D, model.H, model.C, domain=pyo.NonNegativeReals, )

    model.specialistWorkloadsCon = pyo.Constraint(model.DHSC, rule=specialistWorkloadsCon)

    model.webClockRatio = pyo.Param(default=0.8, mutable=True)

    if lp_solve:
        model.webClockRatio = 0.9
        model.centerHourlyBacklog = pyo.Var(model.D, model.H, model.C,
                                            bounds=lambda m, d, h, c: (0, m.centerMaxBacklogSize[d, h, c]),
                                            domain=pyo.NonNegativeReals, )

        model.centerFullTimeUnmetDemand = pyo.Var(model.D, model.mixedC, domain=pyo.NonNegativeReals, )
        model.OBJ = pyo.Objective(rule=objective_lp)

        model.centerDemandBalanceCon = pyo.Constraint(model.D, model.H, model.C, rule=centerDemandBalanceCon)
        model.centerFulltimeCoverCon = pyo.Constraint(model.D, model.mixedC, rule=centerFulltimeWorkCon)

    else:
        model.specialistWorks = pyo.Var(model.S, domain=pyo.Binary, )
        model.specialistAtCenter = pyo.Var(model.SC, domain=pyo.Binary, )

        model.OBJ = pyo.Objective(rule=objective)

        model.specialistAtCenterCon = pyo.Constraint(model.SC, rule=specialistAtCenterCon)
        model.specialistAtCenterByHourCon = pyo.Constraint(model.DHSC, rule=specialistAtCenterByHourCon)
        model.specialistOneCenterPerHourCon = pyo.Constraint(model.DHS, rule=specialistOneCenterPerHourCon)
        model.centerDemandCon = pyo.Constraint(model.D, model.H, model.C, rule=centerDemandCon)

    return model


def objective(model):
    unmetDemand = sum(
        model.centerHourlyUnmetDemand[d, h, c]
        for d in model.D for h in model.H for c in model.C
    )

    preference = sum(
        model.specialistAtCenter[s, c]*model.specialistPreference[s,c]
        for (s, c) in model.SC
    )

    return unmetDemand  + preference

def objective_lp(model):
    unmetDemand = sum(
        100 * model.centerHourlyUnmetDemand[d, h, c]
        for d in model.D for h in model.H for c in model.C
    )

    unmetFulltime= sum(
        10 * model.centerFullTimeUnmetDemand[d, c]
        for d in model.D for c in model.mixedC
    )

    backlog = sum(
        model.centerHourlyBacklog[d, h, c]
        for d in model.D for h in model.H for c in model.C
    )


    return unmetDemand + unmetFulltime +backlog


def specialistAtCenterByHourCon(model, d,h,s,c):
    return model.specialistAtCenterByHour[d,h,s,c] <= model.specialistAtCenter[s,c]

def specialistAtCenterCon(model, s,c):
    return model.specialistAtCenter[s,c] <= model.specialistWorks[s]

def specialistOneCenterPerHourCon(model, d,h,s):
    return sum(model.specialistAtCenterByHour[d, h, s, c] for c in model.C if (d, h, s, c) in model.DHSC) <= 1

def specialistWorkloadsCon(model, d,h,s,c):
    return model.specialistWorkloads[d,h,s,c] <= model.specialistHourlyCapacity[s] * model.webClockRatio * model.specialistAtCenterByHour[d,h,s,c]

def centerDemandBalanceCon(model, d, h, c):
    if d==1 and h == 0:  # d=1,h=0: day7,hour 23
        previous_backlog = model.centerHourlyBacklog[len(model.D), len(model.H) - 1, c]
    elif h==0: #h = 0: d-1,hour 23
        previous_backlog = model.centerHourlyBacklog[d - 1, len(model.H) - 1, c]
    else:
        previous_backlog = model.centerHourlyBacklog[d, h - 1, c]

    return (
            sum(model.specialistWorkloads[d, h, s, c] for s in model.S if (d, h, s, c) in model.DHSC)
            + model.centerHourlyBacklog[d, h, c]
            + model.centerHourlyUnmetDemand[d, h, c]
            ==
            model.centerDemand[d, h, c] + previous_backlog
    )
def centerDemandCon(model, d, h, c):
    return (
            sum(model.specialistWorkloads[d, h, s, c] for s in model.S if (d, h, s, c) in model.DHSC)
            + model.centerHourlyUnmetDemand[d, h, c]
            ==
            model.centerDemand[d, h, c]
    )

def centerFulltimeWorkCon(model,d,c):
    return (sum(model.specialistWorkloads[d, h, s, c] for h in model.mixedH for s in model.FulltimeS if (d, h, s, c) in model.DHSC) >=
    model.centerFullTimeCoverPerc[c]/100 * sum(model.centerDemand[d, h, c] for h in model.H) - model.centerFullTimeUnmetDemand[d, c])

def solve_assignment(D,H,S,C,DHS,SC,input_data,result_data, is_solve=True,lp_solve=False):
    if lp_solve:
        print(f"{datetime.now()} start lp model")
    else:
        print(f"{datetime.now()} start mip model")

    SC_set = set(SC)
    DHSC = {(d, h, s, c) for (d, h, s) in DHS for c in C if (s, c) in SC_set}

    mixedC = list(input_data['centerFullTimeCoverPerc'].keys())

    mixedH = [h for h in H if input_data['hourIsMixed'].get(h) == 1]

    FulltimeS = [s for s in S if input_data['specialistIsFullTime'].get(s) == 1]

    valid_DS = {(dhsc[0], dhsc[2]) for dhsc in DHSC}
    valid_S = {dhsc[2] for dhsc in DHSC}

    start = datetime.now()
    model = assignment_model(lp_solve)

    instance = model.create_instance({
        None: {
            "D": {None: D},
            "H": {None: H},
            "C": {None: C},
            "S": {None: S},
            "DHS": {None: DHS},
            "SC": {None: SC},
            "DHSC": {None: DHSC},
            "mixedC": {None: mixedC},
            "mixedH": {None: mixedH},
            "FulltimeS": {None: FulltimeS},
            "valid_DS": {None: valid_DS},
            "valid_S": {None: valid_S},
            "centerDemand": input_data["centerDemand"],
            "specialistHourlyCapacity": input_data["specialistHourlyCapacity"],
            "specialistPreference": input_data["specialistPreference"],
            "centerMaxBacklogSize": input_data["centerMaxBacklogSize"],
            "centerFullTimeCoverPerc": input_data["centerFullTimeCoverPerc"],
        }
    })
    if lp_solve:
        # fix variable
        {instance.specialistAtCenterByHour[dhsc].fix(result_data["specialistAtCenterByHour"].get(dhsc, 0)) for dhsc in instance.DHSC}

    end = datetime.now()
    print(f"{datetime.now()} finish generating {end - start}")
    if is_solve:
        opt = pyo.SolverFactory("cplex")
        opt.options["mip_tolerances_mipgap"] = 0.05
        opt.solve(instance, tee=False)
        print(f"{datetime.now()} finish solving {datetime.now() - end}")

    return instance

def modify_model(D, H, S, C, DHS, SC, input_data, result_data, is_solve=True, lp_solve=True):

    instance_lp = solve_assignment(D, H, S, C, DHS, SC, input_data, result_data, True, True)

    center_demand_array = np.array([v for v in instance_lp.centerDemand.values()])
    opt = pyo.SolverFactory("cplex")
    ms = 0
    print(f"{datetime.now()} Model update start")
    start = datetime.now()
    for i in range(10):
         scale = 0.75 + i*0.05
         tmp = datetime.now()
         instance_lp.centerDemand.store_values(dict(zip(instance_lp.centerDemand.keys(), center_demand_array*scale)))
         tmp1 = datetime.now()- tmp
         ms += tmp1.total_seconds()
         results = opt.solve(instance_lp, tee=False)
         # instance_lp.solutions.load_from(results)
         # print("Objective value:", pyo.value(instance_lp.OBJ))
    end = datetime.now()
    print(f"{datetime.now()} Model update time: {ms}")
    print(f"{datetime.now()} Model update total time:  {end - start}")


if __name__ == "__main__":
    input_file = 'Data/input.xlsx'
    result_file = 'Data/result.xlsx'
    D,H,C,S,DHS,SC,input_data = load_data(input_file)
    result_data = load_result(result_file)

    # instance_mip = solve_assignment(D, H, S, C, DHS, SC, input_data, result_data, is_solve=True, lp_solve=False)
    instance_lp = solve_assignment(D, H, S, C, DHS, SC, input_data, result_data, is_solve=True, lp_solve=True)
   #
    # ======= modify demand ===================
    center_demand_array = np.array([v for v in instance_lp.centerDemand.values()])
    opt = pyo.SolverFactory("cplex")
    ms = 0
    print(f"{datetime.now()} Model update start")
    start = datetime.now()
    for i in range(10):
        scale = 0.75 + i*0.05
        tmp = datetime.now()
        instance_lp.centerDemand.store_values(dict(zip(instance_lp.centerDemand.keys(), center_demand_array*scale)))
        tmp1 = datetime.now()- tmp
        ms += tmp1.total_seconds()
        results = opt.solve(instance_lp, tee=False)
    end = datetime.now()
    print(f"{datetime.now()} Model update time: {ms}")
    print(f"{datetime.now()} Model update total time:  {end - start}")






