from IPython import embed
import os


infile = open(
    f'{os.getcwd()}/plans/set_18/n_games_459/all_plans_simple.txt', 'r')


schedule = {}
dumper_counter = -1
for line in infile:
    if 'num' in line:
        dumper_counter += 1
        schedule[dumper_counter] = []

    elif 'PARKED' in line:
        schedule[dumper_counter].append('parked')
    elif 'Time' in line:
        plan = line.strip().split('PLAN: ')[-1]
        plan = [(3 - len(node)) * ' ' + node for node in plan.split(', ')]
        plan_new = " => ".join(plan)
        if len(plan_new) != 10:
            embed(header='10')
        try:
            time = line.strip().split('PLAN: ')[0].strip()[
                :-1].split('Time: ')[1]
        except IndexError:
            embed(header='idz')
        schedule[dumper_counter].append([time, plan_new])


text = ''


for i in range(0, len(schedule), 2):
    text += f'             Dumper num: {i:2.0f}            |            Dumper num: {i+1:2.0f}     \n'
    text += '         ----------------------        |        ----------------------\n'
    text += '   Time:                   Plan:       |   Time:                   Plan:\n'
    more = True
    more_counter = 0
    while more:
        first_line = ''
        if len(schedule[i]) > more_counter:
            information = schedule[i][more_counter]
            if information == 'parked':
                first_line = 'Dumper parked'
            else:
                time = information[0]
                plan = information[1]
                first_line = time + ' '*3 + plan

        first_line = ' '*3 + first_line
        first_line += ' ' * (39 - len(first_line)) + '|'

        second_line = ''
        if i + 1 < len(schedule):
            if len(schedule[i + 1]) > more_counter:
                information = schedule[i+1][more_counter]
                if information == 'parked':
                    second_line = 'Dumper parked'
                else:
                    time = information[0]
                    plan = information[1]
                    second_line = time + ' '*3 + plan
        second_line = ' '*3 + second_line
        second_line += ' ' * (34 - len(first_line)) + '\n'

        text += first_line + second_line

        more_counter += 1
        more1 = len(schedule[i]) > more_counter
        if i + 1 < len(schedule):
            more2 = len(schedule[i+1]) > more_counter

        more = more1 or more2

    text += '   -------------------------------------------------------------------------\n\n'

print(text)
embed()
