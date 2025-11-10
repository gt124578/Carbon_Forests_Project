import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import warnings 

#===============================================
#1. Fonctions de bases


#Paramètres initiaux, par défauts
PARAMETRES_INIT={
    'alpha': 0.5,   #Taux de séquestration de base
    'beta': 0.1,   #Taux de respiration Arbres -> Atmosphère
    'gamma': 0.05,  #Taux de litière Arbres -> Sol
    'delta': 0.02,  #Taux respiration Sol -> Atmosphère ET Arbres -> Sol (selon équations)
    'K': 100.0  #Capacité portante des arbres
}

#Conditions initiales par défaut
Y0_INIT=np.array([800.0, 10.0, 50.0]) # [Ca(0), Ct(0), Cs(0)]

#Intervalle de temps par défaut
INTERVALLE_T=(0, 100) # (t_init, t_final)

#Taux de séquestration du carbone dans les arbres.
def S(Ct, alpha, K):
    Ct=max(0, Ct)
    K=max(1e-9, K)
    return alpha*Ct*max(0, 1-Ct/K)

#Fonction f(y, t) pour le système dy/dt = f(y, t).
def f(y, t, params):
    Ca, Ct, Cs=y
    alpha=params.get('alpha', 0.5)
    beta=params.get('beta', 0.1)
    gamma=params.get('gamma', 0.05)
    delta=params.get('delta', 0.02)
    K=params.get('K', 100.0)
    Ct=max(0, Ct)
    Cs=max(0, Cs)
    s_term=S(Ct, alpha, K)
    dCa_dt=-s_term+beta*Ct+delta*Cs
    dCt_dt=s_term-(beta+delta+gamma)*Ct
    dCs_dt=(gamma+delta)*Ct-delta*Cs
    return np.array([dCa_dt, dCt_dt, dCs_dt])


#Jacobien de la fonction f par rapport à y.
def jac_f(y, t, params):
    Ca, Ct, Cs=y
    alpha=params.get('alpha', 0.5)
    beta=params.get('beta', 0.1)
    gamma=params.get('gamma', 0.05)
    delta=params.get('delta', 0.02)
    K=params.get('K', 100.0)
    Ct=max(0, Ct)
    K=max(1e-9, K)
    if Ct<=0:S_prime=alpha
    elif Ct>=K:S_prime=-alpha
    else: S_prime=alpha*(1-2*Ct/K)
    dfA_dCa=0; dfA_dCt=-S_prime+beta; dfA_dCs=delta
    dfT_dCa=0; dfT_dCt=S_prime-(beta+delta+gamma); dfT_dCs=0
    dfS_dCa=0; dfS_dCt=gamma+delta; dfS_dCs=-delta
    return np.array([
        [dfA_dCa, dfA_dCt, dfA_dCs],
        [dfT_dCa, dfT_dCt, dfT_dCs],
        [dfS_dCa, dfS_dCt, dfS_dCs]]).T




#===============================================
# 2. Solveurs Numériques


#Résout dy/dt = f(y, t) avec Euler explicite. Gère la divergence.
def solve_euler_explicite(y0, t_inter, h, params, max_val=1e9):
    t0, T=t_inter
    num_etapes=int(np.ceil((T-t0)/h)) #Ajustement pour inclure T même si T n'est pas un multiple de h
    t_valeurs=np.linspace(t0, t0+num_etapes*h, num_etapes+1)
    if T>t_valeurs[-1]: #Ajouter T si nécessaire
         t_valeurs=np.append(t_valeurs, T)
    t_valeurs=t_valeurs[t_valeurs<=T] #Pour ne pas dépasser T 
    num_etapes=len(t_valeurs)-1 #Mettre à jour num_etapes
    y_valeurs=np.zeros((len(t_valeurs), len(y0)))
    y_valeurs[0,:]=y0
    divergent=False
    for n in range(num_etapes):
        tn=t_valeurs[n]
        yn=y_valeurs[n, :]
        courant_h=t_valeurs[n+1]-t_valeurs[n] #Recalculer h pour le dernier pas si T n'est pas un multiple

        try:
            yn_suivant=yn+courant_h*f(yn, tn, params)
            if np.any(np.abs(yn_suivant)>max_val) or np.any(np.isnan(yn_suivant)):
                 warnings.warn(f"Euler explicite: Divergence détectée à t~{t_valeurs[n+1]:.2f} (h={courant_h:.2e}). Arrêt.")
                 divergent=True
                 y_valeurs[n+1:, :]=np.nan
                 t_valeurs=t_valeurs[:n+2]
                 y_valeurs=y_valeurs[:n+2, :]
                 break
            y_valeurs[n+1, :]=yn_suivant
        except Exception as e:
             warnings.warn(f"Euler explicite: Erreur à t={tn:.2f} (h={courant_h:.2e}): {e}. Arrêt.")
             divergent=True
             y_valeurs[n+1:, :]=np.nan
             t_valeurs=t_valeurs[:n+2]
             y_valeurs=y_valeurs[:n+2, :]
             break
    return t_valeurs, y_valeurs, divergent




#---Fonctions pour solveurs implicites--

def implicite_solver_function(y_new, y_old, h, t_new, params, methode):
    if methode=='euler_implicite':
        return y_new-y_old-h*f(y_new, t_new, params)
    elif methode=='trapezoidal':
        t_old=t_new-h
        return y_new-y_old-0.5*h*(f(y_old, t_old, params)+f(y_new, t_new, params))
    else: raise ValueError("Méthode implicite inconnue")

def implicite_solver_jacobian(y_new, y_old, h, t_new, params, methode):
    I=np.eye(len(y_new))
    J_f_new=jac_f(y_new, t_new, params)
    if methode=='euler_implicite': return I-h*J_f_new
    elif methode=='trapezoidal': return I-0.5*h*J_f_new
    else: raise ValueError("Méthode implicite inconnue")


#-----------------------------------------



#Résout dy/dt = f(y, t) avec Euler implicite ou Trapèze via Newton
def solve_implicite(y0, t_inter, h, params, methode='euler_implicite'):
    t0, T=t_inter
    num_etapes=int(np.ceil((T-t0)/h))
    t_valeurs=np.linspace(t0, t0+num_etapes*h, num_etapes+1)
    if T>t_valeurs[-1]: #Ajouter T si nécessaire
         t_valeurs=np.append(t_valeurs, T)
    t_valeurs=t_valeurs[t_valeurs<=T] #Pour ne pas dépasser T 
    num_etapes=len(t_valeurs)-1 #Mettre à jour num_etapes
    y_valeurs=np.zeros((len(t_valeurs), len(y0)))
    y_valeurs[0, :]=y0
    solver_failed=False

    for n in range(num_etapes):
        t_courant=t_valeurs[n]
        y_courant=y_valeurs[n, :]
        t_suivant=t_valeurs[n+1]
        courant_h=t_suivant-t_courant #h ajusté pour le dernier pas

        solver_args = (y_courant, courant_h, t_suivant, params, methode)

        #Essaye de résoudre le système non linéaire
        try:
             sol=root(implicite_solver_function, y_courant, args=solver_args, jac=implicite_solver_jacobian, method='lm', options={'xtol': 1e-7})
             if not sol.success:
                 warnings.warn(f"{methode}: Échec solveur non linéaire à t={t_suivant:.2f} (h={courant_h:.2e}). Message: {sol.message}")
                 solver_failed=True
                 y_valeurs[n+1:, :]=np.nan
                 t_valeurs=t_valeurs[:n+2]
                 y_valeurs=y_valeurs[:n+2, :]
                 break
             y_valeurs[n+1, :]=sol.x
        except Exception as e:
             warnings.warn(f"{methode}: Erreur pendant la résolution à t={t_suivant:.2f} (h={courant_h:.2e}): {e}")
             solver_failed=True
             y_valeurs[n+1:, :]=np.nan
             t_valeurs=t_valeurs[:n+2]
             y_valeurs=y_valeurs[:n+2, :]
             break

    return t_valeurs, y_valeurs, solver_failed



#===============================================
# 3. Fonctions pour Simulations et Visualisation




#Affiche C_A, C_T, C_S pour une simulation unique.
def plot_unique_sim(t_valeurs, y_valeurs, title, filename=None):
    plt.figure(figsize=(10, 6))
    if np.any(np.isnan(y_valeurs)):
        valid_idx=~np.isnan(y_valeurs[:, 0]) #Si on a des NaN qui sont présents (échec), on trace uniquement les points valides
        t_valeurs=t_valeurs[valid_idx]
        y_valeurs=y_valeurs[valid_idx]
        title+=" (Simulation Échouée/Incomplète)"
    if len(t_valeurs)==0:
            plt.text(0.5, 0.5, "Échec Immédiat", horizontalalignment='center', verticalalignment='center')
            y_valeurs = np.array([[]]).reshape(0, Y0_INIT.shape[0]) # Array vide de bonne dimension pour éviter erreur plot

    if y_valeurs.shape[0]>0:
        plt.plot(t_valeurs, y_valeurs[:, 0], label='$C_A$ (Atmosphère)', color='blue')
        plt.plot(t_valeurs, y_valeurs[:, 1], label='$C_T$ (Arbres)', color='green')
        plt.plot(t_valeurs, y_valeurs[:, 2], label='$C_S$ (Sols)', color='brown')
    else:
        plt.text(0.5, 0.5, "Aucune donnée à afficher", horizontalalignment='center', verticalalignment='center')
    plt.xlabel('Temps')
    plt.ylabel('Quantité de Carbone')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    if filename:
        plt.savefig(filename)
        print(f"Graphique sauvegardé: {filename}")
    plt.show()



#--- Fonctions exécutant les simulations spécifiques ---



#Exécute Euler explicite avec h stable.
def plotrun_euler_explicite_stable(h=0.5, params=PARAMETRES_INIT, y0=Y0_INIT, t_inter=INTERVALLE_T):
    print(f"\n--- Simulation Euler explicite (h={h}) ---")
    t, y, failed=solve_euler_explicite(y0, t_inter, h, params)
    if not failed:
        print("Simulation reussie.")
    else:
        print("La simulation a potentiellement diverge ou échoue.")
    plot_unique_sim(t, y, f'Séquestration du Co2 - Euler explicite (h={h})', f'euler_explicite_h{h}.png')


#Exécute Euler explicite avec h potentiellement instable.
def plotrun_euler_explicite_instable(h=2.5, params=PARAMETRES_INIT, y0=Y0_INIT, t_inter=INTERVALLE_T):
    print(f"\n--- Simulation Euler explicite (h={h}) ---")
    t, y, failed=solve_euler_explicite(y0, t_inter, h, params)
    if not failed:
        print("Simulation réussie (h était peut-être suffisant pour ce T).")
    else:
        print("La simulation a potentiellement divergé ou échoué comme attendu.")
    plot_unique_sim(t, y, f'Séquestration du Co2 - Euler explicite (h={h}) - Potentiellement Instable', f'euler_explicite_h{h}_instable.png')


#Exécute Euler implicite et affiche.
def plotrun_euler_implicite(h=1.0, params=PARAMETRES_INIT, y0=Y0_INIT, t_inter=INTERVALLE_T):
    print(f"\n--- Simulation Euler implicite (h={h}) ---")
    t, y, failed=solve_implicite(y0, t_inter, h, params, methode='euler_implicite')
    if not failed and len(t)>0:
        print("Simulation réussie.")
        print(f"  État final à t={t[-1]:.2f}: CA={y[-1, 0]:.2f}, CT={y[-1, 1]:.2f}, CS={y[-1, 2]:.2f}")
        initial_total=y0[1] + y0[2]
        final_total=y[-1, 1] + y[-1, 2]
        print(f"Séquestration nette (CT+CS): {final_total - initial_total:.2f}")
    else:
        print("La simulation a échoué.")
    plot_unique_sim(t, y, f'Séquestration du Co2 - Euler implicite (h={h})', f'euler_implicite_h{h}.png')


#Exécute Méthode du Trapèze et affiche.
def plotrun_trapezoidal(h=1.0, params=PARAMETRES_INIT, y0=Y0_INIT, t_inter=INTERVALLE_T):
    print(f"\n--- Simulation Méthode du Trapèze (h={h}) ---")
    t, y, failed=solve_implicite(y0, t_inter, h, params, methode='trapezoidal')
    if not failed:
        print("Simulation réussie.")
        print(f"  État final à t={t[-1]:.2f}: CA={y[-1, 0]:.2f}, CT={y[-1, 1]:.2f}, CS={y[-1, 2]:.2f}")
    else:
        print("La simulation a échoué.")
    plot_unique_sim(t, y, f'Séquestration du Co2 - Méthode du Trapèze (h={h})', f'trapezoidal_h{h}.png')





#---Fonctions pour l'analyse de sensibilité de chaque paramètre---



#Fonction générique pour analyser et tracer la sensibilité à un paramètre.
def sensibilite_chaque_param(param_name, param_valeurs, h=0.5, base_params=PARAMETRES_INIT, y0=Y0_INIT, t_inter=INTERVALLE_T,solver_methode='euler_implicite'):
    print(f"\n--- Analyse de sensibilité pour '{param_name}' (méthode: {solver_methode}, h={h}) ---")
    plt.figure(figsize=(12, 7))
    colors=plt.cm.viridis(np.linspace(0, 1, len(param_valeurs)))
    base_value=base_params.get(param_name, 'N/A')
    print(f"Valeur de base pour {param_name}: {base_value}")
    print(f"Valeurs testées: {[f'{v:.3g}' for v in param_valeurs]}")
    for i, p_val in enumerate(param_valeurs):
        param_actuel=base_params.copy()
        if param_name not in param_actuel:
             print(f"Erreur : Paramètre '{param_name}' inconnu.")
             return
        param_actuel[param_name]=p_val
        t_vals, y_vals, failed = solve_implicite(y0, t_inter, h, param_actuel, methode=solver_methode) #Exécute le solveur implicitee choisi
        if failed:
            print(f"  Échec simulation pour {param_name} = {p_val:.3g}")
            continue
        #Tracer C_T et C_S
        if len(t_vals)>0:
            label_suffix=f"{param_name}={p_val:.2e}"
            plt.plot(t_vals, y_vals[:, 1], linestyle='-', color=colors[i], label=f'$C_T$, {label_suffix}')
            plt.plot(t_vals, y_vals[:, 2], linestyle='--', color=colors[i], label=f'$C_S$, {label_suffix}')
    plt.xlabel('Temps')
    plt.ylabel('Quantité de Carbone')
    plt.title(f'Impact de "{param_name}" sur $C_T$ (continu) et $C_S$ (pointillé)')
    plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5)) # On met la légende à l'extérieur pour éviter de masquer les courbes
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) #On ajuste pour la légende
    filename = f"sensitivity_{param_name}_{solver_methode}_h{h}.png"
    plt.savefig(filename)
    print(f"Graphique de sensibilité sauvegardé: {filename}")
    plt.show()





# ===============================================
# 4. Appels des Fonctions (

#---Exécution des solveurs simples---

plotrun_euler_explicite_stable(params={'alpha': 1, 'beta': 0.1, 'gamma': 0.01, 'delta': 0.05, 'K': 300.0},h=0.5)
plotrun_euler_explicite_instable(params={'alpha': 1, 'beta': 0.1, 'gamma': 0.01, 'delta': 0.05, 'K': 300.0},h=2.5)
#plotrun_euler_implicite(h=1)
#plotrun_trapezoidal(h=1.0)



#---Exécution des analyses de sensibilité (une par paramètre)---

#Définir les plages de valeurs pour chaque paramètre
alpha_range=np.linspace(PARAMETRES_INIT['alpha']*0.5, PARAMETRES_INIT['alpha']*1.5, 5)
K_range=np.linspace(PARAMETRES_INIT['K']*0.5, PARAMETRES_INIT['K']*1.5, 5)
beta_range=np.linspace(max(0.001, PARAMETRES_INIT['beta']*0.5), PARAMETRES_INIT['beta']*1.5, 5)
gamma_range=np.linspace(max(0.001, PARAMETRES_INIT['gamma']*0.5), PARAMETRES_INIT['gamma']*1.5, 5)
delta_range=np.linspace(max(0.001, PARAMETRES_INIT['delta']*0.5), PARAMETRES_INIT['delta']*1.5, 5)

#sensibilite_chaque_param('alpha', alpha_range, h=0.5, solver_methode='euler_implicite')
#sensibilite_chaque_param('K', K_range, h=0.5, solver_methode='euler_implicite')
#sensibilite_chaque_param('beta', beta_range, h=0.5, solver_methode='euler_implicite')
#sensibilite_chaque_param('gamma', gamma_range, h=0.5, solver_methode='euler_implicite')
#sensibilite_chaque_param('delta', delta_range, h=0.5, solver_methode='euler_implicite')


#Exemple avec la méthode du Trapèze pour alpha
#sensibilite_chaque_param('alpha', alpha_range, h=0.5, solver_methode='trapezoidal')

print("\n ------- Fin du script -------- \n")
