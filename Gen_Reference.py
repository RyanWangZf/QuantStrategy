# -*- coding: utf-8 -*-
# Generate the Reference with normal format
import readline
readline.parse_and_bind("control-v: paste")

print("###########################################################\n")
print("Generate the normal formatted Reference! (Version 0.1 beta) \n")
print("Author: Ryan Wang\n")
print("Date: 24/6/2018\n")
print("Contact: ryanwang96@hotmail.com \n")
print("###########################################################\n")

if __name__ == "__main__":
    while True:
        components_dict = dict().fromkeys(["title","year","author","journal","vol","no","page"],"")
        counter = 0
        for k,v in components_dict.items():
            if k == "author":
                while True:
                    var = input("Enter {}:".format(k))
                    components_dict[k] += str(var) + ", "
                    var = input("Enter any key to the next item, n for the next author.")
                    if var != "n":
                        break
            else:
                var = input("Enter {}:".format(k))
                components_dict[k] += str(var)
        
        result = components_dict["author"] + "(%s)"%(components_dict["year"]) + ". " + components_dict["title"] + ", " + \
            components_dict["journal"] + ", " + "%s(%s), "%(components_dict["vol"],components_dict["no"]) + components_dict["page"] + "."
        print("Result with normal format:\n",result)
        print("-----------------------------------------------------------\n")
        var = input("Enter any key for next paper, enter 'q' to break out.")
        if var == "q":
            print("Task Complete!")
            break
        
    # example:    
    # title = "Application of Muhlbauer Risk Assessment Method in Pipeline Risk Assessment"
    # year = "2006"
    # author = "Wang K.L., Cao M.M., Wang, B.D."
    # journal = "Research of Environmental Sciences"
    # vol = "19"
    # no = "2"
    # page = "112-114"
    # result = Wang, K.L., Cao, M.M., Wang, B.D., (2006). Application of Muhlbauer Risk Assessment Method in Pipeline Risk Assessment, Research of Environmental Sciences, 19(2), 112-114.
    
    
    
        