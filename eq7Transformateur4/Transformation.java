package abstraction.eq7Transformateur4;

//fichier codé par  Eliott et Pierrick
//Eliott : tout ce qui concerne les journaux
//Pierrick : tout le reste
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import abstraction.eqXRomu.contratsCadres.ExemplaireContratCadre;
import abstraction.eqXRomu.filiere.Filiere;
import abstraction.eqXRomu.general.Journal;
import abstraction.eqXRomu.produits.Chocolat;
import abstraction.eqXRomu.produits.ChocolatDeMarque;
import abstraction.eqXRomu.produits.Feve;

public class Transformation {

	protected Journal journalTransfo ;
	protected List<Double> lescouts; //liste qui contiendra certains des couts à faire pour un step, initialiser à une liste vide à chaque début de l'appel next
	

	public Transformation() {
		//super();
		//this.journalTransfo = new Journal(this.getNom()+" journal transfo", this);
			}
	
	/*
//Pierrick
	public void next() {
		
		super.next();
		this.journalTransfo.ajouter("=== STEP " + Filiere.LA_FILIERE.getEtape() + "==================");
		HashMap<ChocolatDeMarque, Double> chocoalivrer = new HashMap<ChocolatDeMarque,Double>(); //critère pour la transformation
		this.lescouts = new LinkedList<Double>();//on initialise avec une liste vide (supprime les données du step precedent)
		this.coutproduction_tonne_marque_step = new HashMap<ChocolatDeMarque,Double>(); //on la reset à 0, permettra de savoir combien on a dépensé pour fabriquer une tonne d'un chocolat
		this.production_tonne_marque_step = new HashMap<ChocolatDeMarque,Double>();//on la reset à 0, permettre de savoir combien on a produit d'un chocolat ce step
		double peutproduireemploye = this.tauxproductionemploye*this.nbemployeCDI; //ce qu'on peut produire avec nos employés (en terme de conversion de fève)
		double peutencoreproduire = peutproduireemploye; //ce qu'on peut encore produire ce step, au départ c'est tout ce que nos employé peuvent faire
		
		//////////on fixe le critère : ce qu'on doit produire comme choco de marque, dépendant des contrats cadre en cours pour l'instant//////////
		
		//Pour les chocos de marque cocoasis//////////////////////////////////////////////////////////
		for (ChocolatDeMarque c : chocolatCocOasis) {
			double alivrer = 0.0;
			for (ExemplaireContratCadre contrat : this.contratsEnCours) {
				if (contrat.getProduit().equals(c)) {
					alivrer = alivrer + contrat.getQuantiteALivrerAuStep();
				}
			}
			chocoalivrer.put(c, alivrer);
			this.coutproduction_tonne_marque_step.put(c, this.coutmachine + this.coutadjuvant*0.2);//pas oublier l'initialisation de cette Hashmap
			this.production_tonne_marque_step.put(c, 0.0);//idem
		}
		
		//Pour les chocos de marque distributeurs///////////////////////////////////////////////////////
		for (ChocolatDeMarque c : chocolatDistributeur) {
			double alivrer = 0.0;
			for (ExemplaireContratCadre contrat : this.contratsEnCours) {
				if (contrat.getProduit().equals(c)) {
					alivrer = alivrer + contrat.getQuantiteALivrerAuStep();
				}
			}
			chocoalivrer.put(c, alivrer);
			this.coutproduction_tonne_marque_step.put(c, this.coutmachine + this.coutadjuvant*0.2); //initialisation de la hashmap
			this.production_tonne_marque_step.put(c, 0.0);//idem
		}
		//remarquons que chocoalivrer contient en clés tout les chocolat de chocolatCocOasis et de chocolatDistributeur
		
		//////////On fixe ce qu'on produit et on le produit si on en a pas assez en stock//////////
		
		//Pour les chocos de marque cocoasis//////////////////////////////////////////////////////////
		for (ChocolatDeMarque c : chocolatCocOasis) {
			//ça fait la boucle pour tout les chocos de marques
			//On doit quand même savoir quelle fève on va utiliser pour la transformation, ce qui dépend du chocolat produit
			Feve feve_utilise = Feve.F_HQ_BE; //valeur par défault au début, on prend la meilleure fève car elle permet de produire tout les chocolat, de toute façon cette valeur va changer
			//Ici on regarde exhaustivement, c'est un peu lourd mais ça marchera pour n'importe quel chocolat
			if (c.getChocolat() == Chocolat.C_HQ_BE) {
				feve_utilise = Feve.F_HQ_BE; //là on aurait pas vraiment besoin de changer mais c'est pour bien illustrer
			}
			if (c.getChocolat() == Chocolat.C_HQ_E) {
				feve_utilise = Feve.F_HQ_E;
			}
			if (c.getChocolat() == Chocolat.C_HQ) {
				feve_utilise = Feve.F_HQ;
			}
			if (c.getChocolat() == Chocolat.C_MQ_E) {
				feve_utilise = Feve.F_MQ_E;
			}
			if (c.getChocolat() == Chocolat.C_MQ) {
				feve_utilise = Feve.F_MQ;
			}
			if (c.getChocolat() == Chocolat.C_BQ) {
				feve_utilise = Feve.F_BQ;
			}
			//là on sais quelle fève on utilise
			double qtutile1 = 0; //correspond à la qte de fève qu'on va effectivement transformer
			double stock = this.stockFeves.get(feve_utilise);
			double aproduire = 0.0; //la qte de chocolat qu'on va devoir produire
			if (this.stockChocoMarque.get(c) < chocoalivrer.get(c)+5000) { 
				//on veut produire ce qu'on doit livrer et avoir un stock au dessus de 5000 pour pouvoir faire des contrats cadre, on se fixe 5000 comme valeur
				aproduire = aproduire + chocoalivrer.get(c) ;
			}
			//si on a pas le stock nécessaire pour lancer des contrat cadre, on le produit
			if (this.stockChocoMarque.get(c) < 5000) {
					aproduire = aproduire + 2000; //2000 parce qu'on ne peut produire que 17000 (ou autre chose) chaque step, comme ça en 3 tour on peut remettre notre stock au dessus de 5000
			}
			//donc si on a pas assez de chocolat, on va nécessairement produire de quoi respecter les contrats mais aussi de quoi relancer des contrats
			double fevenecessaire = aproduire/(this.pourcentageTransfo.get(feve_utilise).get(c.getChocolat())); //formule conversion entre qte feve et qte choco
			if (stock > 0) {
				if (stock > fevenecessaire) {
				//on a assez en stock, on produit ce qu'on a à produire si on a la main d'oeuvre nécessaire
					if (fevenecessaire < peutencoreproduire) {
						qtutile1 = fevenecessaire;
					} else {
						qtutile1 = peutencoreproduire;
					}
				} else {
				//notre stock est plus petit que ce qu'on devrait utiliser, on utilise tout notre stock si on a la main d'oeuvre nécessaire
					if (stock < peutencoreproduire) {
						qtutile1 = stock;
					} else {
						qtutile1 = peutencoreproduire;
					}
				}
			} //else : on fait rien, on peut pas produire car on a pas de feves
			peutencoreproduire = peutencoreproduire - qtutile1; // on retire de ce qu'on peut produire au total ce qu'on va effectivement produire
			this.stockFeves.replace(feve_utilise, stock - qtutile1);//on retire qtutile1 du stock de feve utilise pour faire du chocolat
			double qtechocoproduit = qtutile1*this.pourcentageTransfo.get(feve_utilise).get(c.getChocolat()); //la qte de choco produit à partir de qtutile1
			
			this.journalTransfo.ajouter("stock de " + c + " avant transfo est "+ this.stockChocoMarque.get(c));
					
			this.stockChocoMarque.replace(c, this.stockChocoMarque.get(c) + qtechocoproduit);
			this.totalStocksChocoMarque.ajouter(this, qtechocoproduit, cryptogramme);
			
			this.journalTransfo.ajouter("stock de " + c + " après transfo est "+ this.stockChocoMarque.get(c));
					
			double payermachine = qtutile1*this.coutmachine; //prix des machines car on transforme une certaine qté de fèves
			double payeradjuvant = this.coutadjuvant*(this.pourcentageTransfo.get(feve_utilise).get(c.getChocolat())-1)*qtutile1; //formule tenant compte du pourcentage d'adjuvant
			this.lescouts.add(payermachine);
			this.lescouts.add(payeradjuvant);
			this.production_tonne_marque_step.replace(c, this.production_tonne_marque_step.get(c)+qtechocoproduit);
			if (qtechocoproduit != 0.0) {
				this.coutproduction_tonne_marque_step.replace(c, (payermachine+payeradjuvant)/qtechocoproduit);
			}
			//on ne va pas prendre en compte le prix des fèves utilisé dans cette transfo, il faudra une marge qui en tient compte
			//on va tenir compte des cout fixe après à la fin des boucle for
		}

		//Pour les chocos de marque distributeurs///////////////////////////////////////////////////////
		for (ChocolatDeMarque c : chocolatDistributeur) {
			this.journalTransfo.ajouter("le chocolat distributeur est " + c);
			//faut réussir à identifier à quel chocolat de notre stock_choco correspond le chocolat de marque distributeur
			
			//On détermine quelle fève on va utiliser pour la transformation, ce qui dépend du chocolat produit
			Feve feve_utilise = Feve.F_HQ_BE; //valeur par défault au début, on prend la meilleure fève car elle permet de produire tout les chocolat, de toute façon cette valeur va changer
			//Ici on regarde exhaustivement, c'est un peu lourd mais ça marchera pour n'importe quel chocolat
			if (c.getChocolat() == Chocolat.C_HQ_BE) {
				feve_utilise = Feve.F_HQ_BE; //là on aurait pas vraiment besoin de changer mais c'est pour bien illustrer
			}
			if (c.getChocolat() == Chocolat.C_HQ_E) {
				feve_utilise = Feve.F_HQ_E;
			}
			if (c.getChocolat() == Chocolat.C_HQ) {
				feve_utilise = Feve.F_HQ;
			}
			if (c.getChocolat() == Chocolat.C_MQ_E) {
				feve_utilise = Feve.F_MQ_E;
			}
			if (c.getChocolat() == Chocolat.C_MQ) {
				feve_utilise = Feve.F_MQ;
			}
			if (c.getChocolat() == Chocolat.C_BQ) {
				feve_utilise = Feve.F_BQ;
			}
			//là on sais quelle fève on utilise
			double qtutile1 = 0; //correspond à la qte de fève qu'on va effectivement transformer
			double stock = this.stockFeves.get(feve_utilise);
			double aproduire = 0.0; //la qte de chocolat qu'on va devoir produire
			if (this.stockChoco.get(c.getChocolat()) < chocoalivrer.get(c)+5000) { 
				//on veut produire ce qu'on doit livrer et avoir un stock au dessus de 25000 pour pouvoir faire des contrats cadre
				aproduire = aproduire + chocoalivrer.get(c) ;
			}
			//si on a pas le stock nécessaire pour lancer des contrat cadre, on le produit
			if (this.stockChoco.get(c.getChocolat()) < 5000) {
					aproduire = aproduire + 2000; //2000 parce qu'on ne peut produire que 17000 (ou autre chose) chaque step, comme ça en 3 tour on peut remettre notre stock au dessus de 5000
			}
			//donc si on a pas assez de chocolat, on va nécessairement produire de quoi respecter les contrats mais aussi de quoi relancer des contrats
			double fevenecessaire = aproduire/(this.pourcentageTransfo.get(feve_utilise).get(c.getChocolat())); //formule conversion entre qte feve et qte choco
			if (stock > 0) {
				if (stock > fevenecessaire) {
				//on a assez en stock, on produit ce qu'on a à produire si on a la main d'oeuvre nécessaire
					if (fevenecessaire < peutencoreproduire) {
						qtutile1 = fevenecessaire;
					} else {
						qtutile1 = peutencoreproduire;
					}
				} else {
				//notre stock est plus petit que ce qu'on devrait utiliser, on utilise tout notre stock si on a la main d'oeuvre nécessaire
					if (stock < peutencoreproduire) {
						qtutile1 = stock;
					} else {
						qtutile1 = peutencoreproduire;
					}
				}
			} //else : on fait rien, on peut pas produire car on a pas de feves
			peutencoreproduire = peutencoreproduire - qtutile1; // on retire de ce qu'on peut produire au total ce qu'on va effectivement produire
			this.stockFeves.replace(feve_utilise, stock - qtutile1);//on retire qtutile1 du stock de feve utilise pour faire du chocolat
			double qtechocoproduit = qtutile1*this.pourcentageTransfo.get(feve_utilise).get(c.getChocolat()); //la qte de choco produit à partir de qtutile1
			
			this.journalTransfo.ajouter("stock de " + c + " avant transfo est "+ this.stockChoco.get(c.getChocolat()));
					
			this.stockChoco.replace(c.getChocolat(), this.stockChoco.get(c.getChocolat()) + qtechocoproduit);
			this.totalStocksChoco.ajouter(this, qtechocoproduit, cryptogramme);
			
			this.journalTransfo.ajouter("stock de " + c + " après transfo est "+ this.stockChoco.get(c.getChocolat()));
					
			double payermachine = qtutile1*this.coutmachine; //prix des machines car on transforme une certaine qté de fèves
			double payeradjuvant = this.coutadjuvant*(this.pourcentageTransfo.get(feve_utilise).get(c.getChocolat())-1)*qtutile1; //formule tenant compte du pourcentage d'adjuvant
			this.lescouts.add(payermachine);
			this.lescouts.add(payeradjuvant);
			this.production_tonne_marque_step.replace(c, this.production_tonne_marque_step.get(c)+qtechocoproduit);
			if (qtechocoproduit != 0.0) {
				this.coutproduction_tonne_marque_step.replace(c, (payermachine+payeradjuvant)/qtechocoproduit);
			}
			//on ne va pas prendre en compte le prix des fèves utilisé dans cette transfo, il faudra une marge qui en tient compte
			//on va tenir compte des cout fixe après à la fin des boucle for
		}
		
		
		
		//////////là on paye//////////
		double qtetotaleproduite = 0.0;
		for (ChocolatDeMarque c : this.production_tonne_marque_step.keySet()) {
			qtetotaleproduite = qtetotaleproduite + this.production_tonne_marque_step.get(c);
		}

		double a_payer = 1000*this.nbemployeCDI + 658; //cout des employes et cout fixe des machines par step 
		//cout fixe d'1 tonne indépendamment de la qualité de la tonne
		double cout_fixe_par_tonne = 0.0;
		//ce code ne marche pas trop, si on produit pas assez sur un step on va avoir un cout_fixe_par_tonne trop élevé.
		//if (qtetotaleproduite != 0) {
		//	cout_fixe_par_tonne = a_payer/qtetotaleproduite;
		//}
		cout_fixe_par_tonne = a_payer/peutproduireemploye;
		for (ChocolatDeMarque c : this.coutproduction_tonne_marque_step.keySet()) {
			this.coutproduction_tonne_marque_step.replace(c, this.coutproduction_tonne_marque_step.get(c) + cout_fixe_par_tonne);
		}

		for (double i : lescouts) {
			a_payer = a_payer + i;
		}
		Filiere.LA_FILIERE.getBanque().payerCout(this, cryptogramme, "CoûtTransformation", a_payer); //on paye tout d'un coup
		this.journalTransfo.ajouter("On paye ce step " + a_payer);
		
		
		//////////Modulation du nbre d'employé////
		//on veut garder au moins 2000 employé, en prendre plus si on a besoin de plus de main d'oeuvre et en licencier si on en a pas besoin
		//on les a déjà payer donc on ne traite pas les couts
		if (qtetotaleproduite<peutproduireemploye*0.25) {
			// on a produit moins d'un quart de ce qu'on pouvait produire, c'est pas rentable d'avoir autant d'employé
			int newnbre = (int)(nbemployeCDI*0.75); // ça va tronquer le nbre si c'est pas un entier
			if (newnbre>2000) {
				int a = nbemployeCDI - newnbre;
				this.journalTransfo.ajouter("On licencie " + a + " employés");
				nbemployeCDI = newnbre;
			} else {
				int a = nbemployeCDI -2000;
				this.journalTransfo.ajouter("On licencie " + a + " employés");
				nbemployeCDI = 2000;
			}
		}//sinon on considère que on a besoin des employés au cas où
		
		if(qtetotaleproduite==peutproduireemploye) {
			this.journalTransfo.ajouter("On embauche " + 30 + " employés");
			nbemployeCDI = nbemployeCDI + 30; //on ne peut pas embaucher plus de 30 personnes par step
		}
		this.journalTransfo.ajouter("Notre nombre d'employé est " + nbemployeCDI);
		
		
		//TEST :
		for (Feve f : this.lesFeves) {
			this.journalTransfo.ajouter("stock de fève" + f+ "après ce step est "+ this.stockFeves.get(f));
		}
		
		for (ChocolatDeMarque c : this.coutproduction_tonne_marque_step.keySet()) {
			this.journalTransfo.ajouter("Quantite produite de " + c + " à ce step : "+ this.production_tonne_marque_step.get(c));
		}
		


	}
	
		public List<Journal> getJournaux() {
			List<Journal> res=super.getJournaux();
			res.add(journalTransfo);
			return res;
		}
		
		*/
		
}
