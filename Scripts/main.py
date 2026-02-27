# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Paris Brain Institute. All rights reserved.

Created on January 2026

@author: Cassandra Dumas

"""
# Modules
# -------

import os
os.chdir((os.path.dirname(os.path.realpath(__file__))))


# Functions
# ---------
def main():
    print("\n_______________________________")
    print("Inter Expert Variability Evaluation")
    print("_______________________________\n")
    
    
    commands = {
        1 : "Inter-expert agreement (stats)",
        2 : "CSP Selection Variability (stats)",
        3 : "Expert Influence (stats)",
        4 : "Paper Figures",
        5 : "Additionnal Figures"
        }
    
    
    # Input
    # -----
    print("\nIn the list above, you will find the different analyses available")
    for cle, valeur in commands.items():
        print(str(cle) + " : " + valeur)
    key = int(input("\nSelect the analysis you want to perform \n"))
    analysis = commands[key]
    
    ####### AGREEMENT #######
    # -----------------------
    if analysis == "Inter-expert agreement (stats)":
        from Inter_Expert_Agreement import inter_expert_agreement_stats
        inter_expert_agreement_stats()
    
    ####### CSP SELECTION VARIABILITY #######
    # ---------------------------------------
    if analysis == "CSP Selection Variability (stats)":
        from CSP_Selection_Variability import CSP_Variability_Stats
        CSP_Variability_Stats()
        
    ####### EXPERT INFLUENCE #######
    # ---------------------------------------
    if analysis == "Expert Influence (stats)":
        from Expert_Influence import Expert_Influence_Analysis
        Expert_Influence_Analysis()
        
    ####### PAPER FIGURES #######
    # ---------------------------
    if analysis == "Paper Figures":
        from Paper_Figures import Make_Paper_Figures
        Make_Paper_Figures()
        
    ####### ADDITIONNAL FIGURES #######
    # ---------------------------------
    if analysis == "Additionnal Figures":
        from Paper_Figures_Additional import Make_Additional_Figures
        Make_Additional_Figures()
        
        
####### MAIN #######
# ------------------
if __name__ == '__main__':
    main()