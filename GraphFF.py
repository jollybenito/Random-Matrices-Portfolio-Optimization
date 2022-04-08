# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:31:59 2022

@author: benit
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
              
#----------------------------------------------------
# Frontier graphs made function for proper implementation
def frontier_graph( VAR_in2, VAR_in1, VAR_out2, VAR_out1,
                         name,name2, mu=None):
    """
        Create efficient frontier graph and calculate the risks for the Markowitz
        portfolio.
        Args:
            Cov_in1 (np.array): Covariance in-sample built with modified eigenvalues.
            Cov_in2 (np.array): Covariance in-sample built with the original eigenvalues.
            Cov_out1 (np.array): Covariance out-sample built with modified eigenvalues.
            Cov_out2 (np.array): Covariance out-sample built with the original eigenvalues.
            name (string): Name of the database.
            name2 (string): Name of the method used to clean the eigenvalues.
            mu (Optional)(np.array): Mean vector of the database.
        Returns:
            Png file with the efficient frontier graph.
            Tuple containing:
                VAR_in1 (np.array): Estimated risks in-sample built with modified eigenvalues.
                VAR_in2 (np.array) : Estimated risks out-sample built with the original eigenvalues.
                VAR_out1 (np.array): Estimated risks out-sample built with modified eigenvalues.
                VAR_out2  (np.array) : Estimated risks out-sample built with the original eigenvalues.
    """
    p=VAR_in1.shape[0]  
    if mu is None: mu = np.ones(p)
    points = 100
    G = np.linspace(0.01,100,points)
    Metrics_tab = np.zeros([2,3])
    ### IN-SAMPLE
    #clean    
    plt.scatter(VAR_in1, G, label=r'$\xi_i$ (in-sample)', alpha=0.5, s = 5) #'Clean(in-sample)'                    
    ### OUT-SAMPLE
    #clean    
    plt.scatter(VAR_out1, G, label=r'$\xi_i$ (out-sample)', alpha=0.5, s = 5) #'Clean(out-sample)'                
    #dirty    
    plt.scatter(VAR_in2, G, label=r'$\lambda_i$ (in-sample)', alpha=0.5, s = 5) #'Dirty(in-sample)'                    
    #dirty    
    plt.scatter(VAR_out2, G, label=r'$\lambda_i$ (out-sample)', alpha=0.5, s = 5) #'Dirty(out-sample)'                
    # plot specifications
    plt.ylim([0,100])
    plt.legend(loc='lower left',bbox_to_anchor = (0.5, 0.2))   
    plt.xlabel(r'$\mathcal{R}^2$') 
    plt.ylabel(r'$\mathcal{G}$')
    plt.savefig(name + '_Markowitz_' + name2 + '.png',dpi=300)
    # plt.show()
    plt.clf()
    
    Metrics_tab=pd.DataFrame(Metrics_tab)
    Metrics_tab.columns=["Razón de Sharpe","Razón de Treynor","Razón de Alpha"]
    Metrics_tab.index = [r"$\xi_i$ In_" + name2,r"$\lambda_i$ In" + name2]
    
    return (VAR_in1, VAR_in2, VAR_out1, VAR_out2, Metrics_tab)
#---------------------------------------------------
#----------------------------------------------------
# Frontier graphs OVERLAYED
def frontier_graph_over(VAR_in2, VAR_in1, VAR_out2, VAR_out1,
                        VAR_in3, VAR_in4, VAR_out3, VAR_out4,
                        VAR_in5, VAR_out5, name,name2, mu=None):
    """
        Create efficient frontier graph and calculate the risks for the Markowitz
        portfolio.
        Args:
            Cov_in1 (np.array): Covariance in-sample built with modified eigenvalues.
            Cov_in2 (np.array): Covariance in-sample built with the original eigenvalues.
            Cov_out1 (np.array): Covariance out-sample built with modified eigenvalues.
            Cov_out2 (np.array): Covariance out-sample built with the original eigenvalues.
            name (string): Name of the database.
            name2 (string): Name of the method used to clean the eigenvalues.
            mu (Optional)(np.array): Mean vector of the database.
        Returns:
            Png file with the efficient frontier graph.
            Tuple containing:
                VAR_in1 (np.array): Estimated risks in-sample built with modified eigenvalues.
                VAR_in2 (np.array) : Estimated risks out-sample built with the original eigenvalues.
                VAR_out1 (np.array): Estimated risks out-sample built with modified eigenvalues.
                VAR_out2  (np.array) : Estimated risks out-sample built with the original eigenvalues.
    """
    string3=name+".csv"
    X =  pd.read_csv(string3)
    # Eliminates first column, because it's the index.
    return_vec = X.drop(X.columns[[0]], axis=1)     
    t1,t2=np.shape(return_vec)
    if t1>t2: return_vec = np.transpose(return_vec) 
    p, n = np.shape(return_vec)
    if mu is None: mu = np.ones(p)
    points = 100
    G = np.linspace(0.01,100,points)
    Metrics_tab = np.zeros([2,3])
    ### IN-SAMPLE
    #clean    
    plt.scatter(VAR_in1, G, label=r'$Recorte$',  c="r", alpha=0.5, s = 5) #'Clean(in-sample)'                    
    ### OUT-SAMPLE
    #clean    
    plt.scatter(VAR_out1, G,  c="r", alpha=0.5, s = 5) #'Clean(out-sample)'                
    #clean    
    plt.scatter(VAR_in3, G, label=r'$TW$', c="blue", alpha=0.5, s = 5) #'Clean(in-sample)'                    
    #dirty    
    plt.scatter(VAR_in4, G, label=r'$Contracción \ lineal$', c="orange", alpha=0.5, s = 5) #'Dirty(in-sample)'                    
    ### OUT-SAMPLE
    #clean    
    plt.scatter(VAR_out3, G,  c="blue", alpha=0.5, s = 5) #'Clean(out-sample)'                
    #dirty    
    plt.scatter(VAR_out4, G, c="orange", alpha=0.5, s = 5) #'Dirty(out-sample)'                
    #clean    
    plt.scatter(VAR_in5, G, label=r'$Contracción \ no \ lineal$', c="green", alpha=0.5, s = 5) #'Dirty(in-sample)'                    
    ### OUT-SAMPLE
    #clean    
    plt.scatter(VAR_out5, G, c="green", alpha=0.5, s = 5) #'Dirty(out-sample)'                 
    #dirty    
    plt.scatter(VAR_in2, G, label=r'$Original$',  c="black", alpha=0.5, s = 5) #'Dirty(in-sample)'                    
    #dirty    
    plt.scatter(VAR_out2, G, c="black", alpha=0.5, s = 5) #'Dirty(out-sample)'                



    # plot specifications
    plt.ylim([0,100])
    plt.legend(loc='lower left',bbox_to_anchor = (0.5, 0.2))   
    plt.xlabel(r'$\mathcal{R}^2$') 
    plt.ylabel(r'$\mathcal{G}$')
    plt.savefig(name + name2 + '.png',dpi=300)
    # plt.show()
    plt.clf()
    
    Metrics_tab=pd.DataFrame(Metrics_tab)
    Metrics_tab.columns=["Razón de Sharpe","Razón de Treynor","Razón de Alpha"]
    Metrics_tab.index = [r"$\xi_i$ In_" + name2,r"$\lambda_i$ In" + name2]
    
    return (VAR_in1, VAR_in2, VAR_out1, VAR_out2, Metrics_tab)
#----------------------------------------------------
# Frontier graphs made function for proper implementation
def graph_main(name):
    VAR_ = pd.read_csv(name + "VAR_Portfolio.csv")
    VAR_2 = VAR_.drop(columns = "Unnamed: 0")
    VAR__0 = VAR_2.iloc[0,:]
    VAR__1 = VAR_2.iloc[1,:] 
    VAR__2 = VAR_2.iloc[2,:]
    VAR__3 = VAR_2.iloc[3,:]
    VAR__4 = VAR_2.iloc[4,:]
    VAR__5 = VAR_2.iloc[5,:]
    VAR__6 = VAR_2.iloc[6,:]
    VAR__7 = VAR_2.iloc[7,:]
    VAR__8 = VAR_2.iloc[8,:]
    VAR__9 = VAR_2.iloc[9,:]
    frontier_graph(VAR__0, VAR__2, VAR__1, VAR__3,
                             name,"M_Recorte")
    frontier_graph(VAR__0, VAR__4, VAR__1, VAR__5,
                             name,'M_TW')
    frontier_graph(VAR__0, VAR__6, VAR__1, VAR__7,
                             name,'M_Lineal')     
    frontier_graph(VAR__0, VAR__8, VAR__1, VAR__9,
                   name,'M_No-Lineal')   
    frontier_graph_over(VAR__0, VAR__2, VAR__1, VAR__3,
                   VAR__4, VAR__5, VAR__6, VAR__7,
                   VAR__8, VAR__9,
                   name,"M_Comparativo_v2")

    VAR_ = pd.read_csv(name + "VARCluster_Portfolio.csv")
    VAR_2 = VAR_.drop(columns = "Unnamed: 0")
    VAR__0 = VAR_2.iloc[0,:]
    VAR__1 = VAR_2.iloc[1,:] 
    VAR__2 = VAR_2.iloc[2,:]
    VAR__3 = VAR_2.iloc[3,:]
    VAR__4 = VAR_2.iloc[4,:]
    VAR__5 = VAR_2.iloc[5,:]
    VAR__6 = VAR_2.iloc[6,:]
    VAR__7 = VAR_2.iloc[7,:]
    VAR__8 = VAR_2.iloc[8,:]
    VAR__9 = VAR_2.iloc[9,:]

    frontier_graph(VAR__0, VAR__2, VAR__1, VAR__3,
                             name,"HRP_Recorte")
    frontier_graph(VAR__0, VAR__4, VAR__1, VAR__5,
                             name,'HRP_TW')
    frontier_graph(VAR__0, VAR__6, VAR__1, VAR__7,
                             name,'HRP_Lineal')     
    frontier_graph(VAR__0, VAR__8, VAR__1, VAR__9,
                   name,'HRP_No-Lineal')   
    frontier_graph_over(VAR__0, VAR__2, VAR__1, VAR__3,
                   VAR__4, VAR__6, VAR__5, VAR__7,
                   VAR__8, VAR__9,
                   name,"HRP_Comparativo_v2", mu=None)
    return()

name="Data_Storage/Synth200_q2_50s/Synthetic200_q2_50s_99"
graph_main(name)
