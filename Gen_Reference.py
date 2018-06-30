# -*- coding: utf-8 -*-
# Generate the Reference with normal format
import readline
import time
readline.parse_and_bind("control-v: paste")

print("###########################################################\n")
print("Generate the normal formatted Reference! (Version 0.1 beta) \n")
print("Author: Ryan Wang\n")
print("Date: 24/6/2018\n")
print("Contact: ryanwang96@hotmail.com \n")
print("###########################################################\n")

def print_help():
    print("-----------------------------------------------------------\n")
    print("# example:\n") 
    print('# title = "Application of Muhlbauer Risk Assessment Method in Pipeline Risk Assessment"\n')
    print('# year = "2006"\n')
    print('# author = "Wang K.L., Cao M.M., Wang, B.D."\n')
    print('# journal = "Research of Environmental Sciences"\n')
    print('# vol = "19"\n')
    print('# no = "2"\n')
    print('# page = "112-114"\n')
    print('# result = Wang, K.L., Cao, M.M., Wang, B.D., (2006). Application of Muhlbauer Risk Assessment Method in Pipeline Risk Assessment, Research of Environmental Sciences, 19(2), 112-114.\n')
    print("-----------------------------------------------------------\n")
    
if __name__ == "__main__":
    var = input("Enter any key to continue, enter 'h' for help.")
    if var  == "h":
        while True:
            print_help()
            var = input("Enter any key to continue, 'h' for help once again.")
            if var != "h":
                break
    while True:
        components_dict = dict().fromkeys(["title","year","author","journal","vol","no","page"],"")
        counter = 0
        for k,v in components_dict.items():
            if k == "author":
                while True:
                    var = input("Enter {}:".format(k))
                    components_dict[k] += str(var) + ", "
                    var = input("Enter any key to the next item, 'n' for input one more author.")
                    if var != "n":
                        break
            else:
                var = input("Enter {}:".format(k))
                components_dict[k] += str(var)
        
        result = components_dict["author"] + "(%s)"%(components_dict["year"]) + ". " + components_dict["title"] + ", " + \
            components_dict["journal"] + ", " + "%s(%s), "%(components_dict["vol"],components_dict["no"]) + components_dict["page"] + "."
        print("Result with normal format:\n",result)
        print("-----------------------------------------------------------\n")
        var = input("Enter any key for next paper, enter 'q' to break out, enter 'h' for help once again.")
        
        if var == "q":
            print("Task Complete! Thank YOU.")
            time.sleep(3)
            break
        elif var == "h":
            while True:
                print_help()
                var = input("Enter any key to continue, 'h' for help once again.")
                if var != "h":
                    break

    
    
    
        