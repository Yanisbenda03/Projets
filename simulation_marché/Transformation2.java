package abstraction.eq7Transformateur4;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import abstraction.eqXRomu.contratsCadres.ExemplaireContratCadre;
import abstraction.eqXRomu.filiere.Filiere;
import abstraction.eqXRomu.general.Journal;
import abstraction.eqXRomu.produits.Chocolat;
import abstraction.eqXRomu.produits.ChocolatDeMarque;
import abstraction.eqXRomu.produits.Feve;

public class Transformation2 extends Transformateur4AcheteurBourse {

	protected Journal journalTransfo ;
	protected List<Double> lescouts; //liste qui contiendra certains des couts à faire pour un step, initialiser à une liste vide à chaque début de l'appel next


	public Transformation2() {
		super();
		this.journalTransfo = new Journal(this.getNom()+" journal transfo", this);
			}
	
	

	public void next() {
		
		super.next();
		this.journalTransfo.ajouter("=== STEP " + Filiere.LA_FILIERE.getEtape() + "==================");
		//HashMap<ChocolatDeMarque, Double> chocoalivrer = new HashMap<ChocolatDeMarque,Double>(); //critère pour la transformation
		
	
		HashMap<ChocolatDeMarque, Double> partProduite = new HashMap<ChocolatDeMarque,Double>(); 
		
		for (Feve f : this.lesFeves) {
			this.journalTransfo.ajouter("stock de fève" + f+ "avant ce step est "+ this.stockFeves.get(f));
		}
		
		
		this.lescouts = new LinkedList<Double>();//on initialise avec une liste vide (supprime les données du step precedent)
		this.coutproduction_tonne_marque_step = new HashMap<ChocolatDeMarque,Double>(); //reset à 0, permettra de savoir combien on a dépensé pour fabriquer une tonne d'un chocolat
		this.production_tonne_marque_step = new HashMap<ChocolatDeMarque,Double>();//reset à 0, permettre de savoir combien on a produit d'un chocolat ce step
		this.peutproduireemploye = this.tauxproductionemploye*this.nbemployeCDI; //ce qu'on peut produire avec nos employés (en terme de conversion de fève)
		
		//Pour les chocos de marque cocoasis//////////////////////////////////////////////////////////
		
		for (ChocolatDeMarque c : chocolatCocOasis) {
			this.coutproduction_tonne_marque_step.put(c, this.coutmachine + this.coutadjuvant*0.2);//initialisation Hashmap
			this.production_tonne_marque_step.put(c, 0.0);//idem
		}
		
		//Pour les chocos de marque distributeurs///////////////////////////////////////////////////////
		for (ChocolatDeMarque c : chocolatDistributeur) {
			this.coutproduction_tonne_marque_step.put(c, this.coutmachine + this.coutadjuvant*0.2); //initialisation Hashmap
			this.production_tonne_marque_step.put(c, 0.0);//idem
		}
	
			
		//#################################### QUANTITE PRODUITE ####################################################
			

		for (ChocolatDeMarque c : this.chocosProduits) {
			if (chocoAProduire.get(c) != 0) {
				partProduite.put(c, chocoAProduire.get(c)/this.totalBesoin);
				
				
				
			} else {
				partProduite.put(c,0.0);
			}
			
			journalTransfo.ajouter("" + c + " prend " + partProduite.get(c) + " de la production");
		}
		
		if (this.totalBesoin < this.peutproduireemploye )	{
			for (ChocolatDeMarque c : this.chocosProduits) {
				for (Feve f : this.lesFeves) {
					if ((c.getGamme().equals(f.getGamme())) && (c.isBio()== f.isBio()) && (c.isEquitable()== f.isEquitable() )) {
						if (feveNecessaire.get(f) <= this.stockFeves.get(f)) {						
							chocoEffectivementProduit.put(c, chocoAProduire.get(c)); 
						} else { 
							double PartPourCeType = 1.0; //représente la part de ce chocolat par rapport à la totalité des chocolats de ce type à produire
							double partTotale = partProduite.get(c);	
							for (ChocolatDeMarque c2 : this.chocosProduits) {
								if (c.getChocolat().equals(c2.getChocolat())  && (!(c.equals(c2)))) {
									partTotale += partProduite.get(c2);
								}
							}
							PartPourCeType = partProduite.get(c) / partTotale;
							
							journalTransfo.ajouter("" + PartPourCeType + " part de " + c);
							chocoEffectivementProduit.put(c, PartPourCeType*this.stockFeves.get(f) *(this.pourcentageTransfo.get(f).get(c.getChocolat())) );
						}
					}	
				}
			}
		} else {
			for (ChocolatDeMarque c : this.chocosProduits) {
				for (Feve f : this.lesFeves) {
					if ((c.getGamme().equals(f.getGamme())) && (c.isBio()== f.isBio()) && (c.isEquitable()== f.isEquitable() )) {
						if (feveNecessaire.get(f) <= this.stockFeves.get(f)) {
							chocoEffectivementProduit.put(c, this.peutproduireemploye * partProduite.get(c)); 
						} else { 
							double PartPourCeType = 1.0; //représente la part de ce chocolat par rapport à la totalité des chocolats de ce type à produire
							double partTotale = partProduite.get(c);		
							for (ChocolatDeMarque c2 : this.chocosProduits) {
								if ((c.getChocolat().equals(c2.getChocolat()) && (!(c.equals(c2))))) {
									partTotale += partProduite.get(c2);
								}
							}
							PartPourCeType = partProduite.get(c) / partTotale;
							
							journalTransfo.ajouter("" + PartPourCeType + " part de " + c);
							chocoEffectivementProduit.put(c,  Math.min(PartPourCeType*this.stockFeves.get(f) *(this.pourcentageTransfo.get(f).get(c.getChocolat())) * partProduite.get(c) , peutproduireemploye * partProduite.get(c)) );
						}
					}	
				}
			}
		}
		
		
		//##################################### PRODUCTION ######################################################
			
		for (ChocolatDeMarque c : chocolatCocOasis) {
			this.stockChocoMarque.put(c, this.stockChocoMarque.get(c) + chocoEffectivementProduit.get(c));
			this.totalStocksChocoMarque.ajouter(this, chocoEffectivementProduit.get(c), cryptogramme);
			
			for (Feve f : this.lesFeves) {
				if ((c.getGamme() == f.getGamme()) && (c.isBio() == f.isBio()) && (c.isEquitable() == f.isEquitable())) {
					
					this.stockFeves.replace(f, this.stockFeves.get(f) - chocoEffectivementProduit.get(c)/(this.pourcentageTransfo.get(f).get(c.getChocolat())));
					this.totalStocksFeves.retirer(this,chocoEffectivementProduit.get(c) /(this.pourcentageTransfo.get(f).get(c.getChocolat())), this.cryptogramme);
			
					double payermachine = chocoEffectivementProduit.get(c)/(this.pourcentageTransfo.get(f).get(c.getChocolat()))*this.coutmachine; //prix des machines car on transforme une certaine qté de fèves
					double payeradjuvant = this.coutadjuvant*0.2*chocoEffectivementProduit.get(c); //formule tenant compte du pourcentage d'adjuvant
					this.lescouts.add(payermachine);
					this.lescouts.add(payeradjuvant);
					this.production_tonne_marque_step.replace(c, this.production_tonne_marque_step.get(c)+chocoEffectivementProduit.get(c));
					if (chocoEffectivementProduit.get(c) != 0.0) {
						this.coutproduction_tonne_marque_step.replace(c, (payermachine+payeradjuvant)/chocoEffectivementProduit.get(c));
					}
				}
			}	
		}
		
		for (ChocolatDeMarque c : chocolatDistributeur) {
			this.stockChoco.put(c.getChocolat(), this.stockChoco.get(c.getChocolat()) + chocoEffectivementProduit.get(c));
			this.totalStocksChoco.ajouter(this, chocoEffectivementProduit.get(c), cryptogramme);
			
			
			for (Feve f : this.lesFeves) {
				if ((c.getGamme() == f.getGamme()) && (c.isBio() == f.isBio()) && (c.isEquitable() == f.isEquitable())) {
					
					this.stockFeves.replace(f, this.stockFeves.get(f) - chocoEffectivementProduit.get(c)/(this.pourcentageTransfo.get(f).get(c.getChocolat())));
					this.totalStocksFeves.retirer(this,chocoEffectivementProduit.get(c) /(this.pourcentageTransfo.get(f).get(c.getChocolat())), this.cryptogramme);
			
					double payermachine = chocoEffectivementProduit.get(c)/(this.pourcentageTransfo.get(f).get(c.getChocolat()))*this.coutmachine; //prix des machines car on transforme une certaine qté de fèves
					double payeradjuvant = this.coutadjuvant*0.2*chocoEffectivementProduit.get(c); //formule tenant compte du pourcentage d'adjuvant
					this.lescouts.add(payermachine);
					this.lescouts.add(payeradjuvant);
					this.production_tonne_marque_step.replace(c, this.production_tonne_marque_step.get(c)+chocoEffectivementProduit.get(c));
					if (chocoEffectivementProduit.get(c) != 0.0) {
						this.coutproduction_tonne_marque_step.replace(c, (payermachine+payeradjuvant)/chocoEffectivementProduit.get(c));
					}
				}
			}	
		}
		
		
		
		//############################################### PAYEMENT #################################################
		
		double qtetotaleproduite = 0.0;
		for (ChocolatDeMarque c : chocosProduits) {
			qtetotaleproduite += chocoEffectivementProduit.get(c);
		}

		double a_payer = 1000*this.nbemployeCDI + 658; //cout des employes et cout fixe des machines par step 
		//cout fixe d'1 tonne indépendamment de la qualité de la tonne
		double cout_fixe_par_tonne = 0.0;
		//ce code ne marche pas trop, si on produit pas assez sur un step on va avoir un cout_fixe_par_tonne trop élevé.
		//if (qtetotaleproduite != 0) {
		//	cout_fixe_par_tonne = a_payer/qtetotaleproduite;
		//}
		cout_fixe_par_tonne = a_payer/this.peutproduireemploye;
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
		if (qtetotaleproduite<this.peutproduireemploye*0.25) {
			if (Filiere.LA_FILIERE.getEtape() > 10) {
				// on a produit moins d'un quart de ce qu'on pouvait produire, c'est pas rentable d'avoir autant d'employé
				int newnbre = (int)(nbemployeCDI*0.90); // ça va tronquer le nbre si c'est pas un entier
				if (newnbre>2000) {
					int a = nbemployeCDI - newnbre;
					this.journalTransfo.ajouter("On licencie " + a + " employés");
					nbemployeCDI = newnbre;
				} else {
					int a = nbemployeCDI -2000;
					this.journalTransfo.ajouter("On licencie " + a + " employés");
					nbemployeCDI = 2000;
				}
			}
		}

			
		//sinon on considère que on a besoin des employés au cas où
		if(qtetotaleproduite>=this.peutproduireemploye) {
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
		
		journalTransfo.ajouter(" notre capacité de transformation est de : " + this.peutproduireemploye);
		journalTransfo.ajouter("total besoin" + this.totalBesoin);
		journalTransfo.ajouter(" nous avons transformé : " + qtetotaleproduite);


	}
	
		public List<Journal> getJournaux() {
			List<Journal> res=super.getJournaux();
			res.add(journalTransfo);
			return res;
		}
		
}
