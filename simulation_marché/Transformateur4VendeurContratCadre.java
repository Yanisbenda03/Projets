package abstraction.eq7Transformateur4;






import java.awt.Color;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import abstraction.eqXRomu.bourseCacao.BourseCacao;
import abstraction.eqXRomu.contratsCadres.Echeancier;
import abstraction.eqXRomu.contratsCadres.ExemplaireContratCadre;
import abstraction.eqXRomu.contratsCadres.IAcheteurContratCadre;
import abstraction.eqXRomu.contratsCadres.IVendeurContratCadre;
import abstraction.eqXRomu.contratsCadres.SuperviseurVentesContratCadre;
import abstraction.eqXRomu.filiere.Filiere;
import abstraction.eqXRomu.general.Journal;
import abstraction.eqXRomu.produits.Chocolat;
import abstraction.eqXRomu.produits.ChocolatDeMarque;
import abstraction.eqXRomu.produits.Feve;
import abstraction.eqXRomu.produits.IProduit;

public class Transformateur4VendeurContratCadre extends Transformateur4AcheteurContratCadre implements IVendeurContratCadre {
	
	private IAcheteurContratCadre acheteurPrecedent;
	private SuperviseurVentesContratCadre supCC;
	

	protected Journal journalVCC;
	private HashMap<Long, Double> prixPrecedentC;

	public Transformateur4VendeurContratCadre() {
		super();
		this.journalVCC = new Journal(this.getNom()+" journal CC vente", this);
		this.prixPrecedentC = new HashMap<Long,Double>();
	}
	
	//////////////je me permet de rajouter ça sinon on initialise pas supCC
	public void initialiser() {
		super.initialiser();
		this.supCC = (SuperviseurVentesContratCadre)(Filiere.LA_FILIERE.getActeur("Sup.CCadre"));
	}
	//////////////
	
	//codé par Eliott		
	public boolean vend(IProduit produit) { 

		for (ChocolatDeMarque c : this.chocolatCocOasis) {
			//System.out.println("   ---"+c+" "+(produit.getType().equals("ChocolatDeMarque"))+" "+((produit.getType().equals("ChocolatDeMarque"))?(c.equals((ChocolatDeMarque)produit)):"...")+" "+((produit.getType().equals("ChocolatDeMarque")&&(c.equals((ChocolatDeMarque)produit)))?(stockChocoMarque.get(produit)>25000):"..."));
			if ((produit.getType().equals("ChocolatDeMarque")) && (c.equals((ChocolatDeMarque)produit)) && (stockChocoMarque.get(produit)> 5000) ) {
				//System.out.println(" on vend "+produit);
				return true;
			}
		}
		for (ChocolatDeMarque c : this.chocolatDistributeur) {
			if ((produit.getType().equals("ChocolatDeMarque")) &&(c.equals((ChocolatDeMarque)produit)) && (stockChoco.get(((ChocolatDeMarque)produit).getChocolat()) > 5000 )) {
				//System.out.println(" on vend "+produit);
				return true;
			}
		}
		//System.out.println(" on ne vend pas "+produit);
		return false ; 
		
		//à modifier selon ce qu'on veut vendre et dans quelles circonstances
	}

	
	
	
	
	//Négociations
	
	public Echeancier contrePropositionDuVendeur(ExemplaireContratCadre contrat) {
		///à modifier selon comment on veut nos échéanciers
		if (!(contrat.getProduit().getType().equals("ChocolatDeMarque"))) {
			return null;
		}
		
		ChocolatDeMarque cM = (ChocolatDeMarque)(contrat.getProduit());

		//si ce sont des mirages , codé par Anaïs et Pierrick
		if ( ((ChocolatDeMarque)contrat.getProduit()).getMarque() == "Mirage") {
			if ((stockChocoMarque.get(cM)-restantALivrerAuStep(cM)-contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances()> 5000) 
			&& ( this.totalBesoin+contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances() < this.peutproduireemploye * 1.0)
			&& ((contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances())<=2000)) {
				if (contrat.getEcheancier().getStepFin()-contrat.getEcheancier().getStepDebut()<11
				|| contrat.getEcheancier().getStepDebut()-Filiere.LA_FILIERE.getEtape()>8) {
					return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, contrat.getEcheancier().getQuantiteTotale()/12 );
				} else { // les volumes sont corrects, la duree et le debut aussi
					return contrat.getEcheancier();
				}
			} else if ((stockChocoMarque.get(cM)-restantALivrerAuStep(cM)-contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances()> 5000) 
			&& ( this.totalBesoin+contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances() < this.peutproduireemploye * 1.0)
			&& ((contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances())>2000)) {
				if (contrat.getEcheancier().getStepFin()-contrat.getEcheancier().getStepDebut()<11
				|| contrat.getEcheancier().getStepDebut()-Filiere.LA_FILIERE.getEtape()>8) {
					return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, 2000 );
				} else { // les volumes sont corrects, la duree et le debut aussi
					return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, contrat.getEcheancier().getNbEcheances(), 2000 );
				}
			} else { //les volumes sont incorrects
				if ((stockChocoMarque.get(cM)-restantALivrerAuStep(cM) - 1000> 5000)
				&& ( this.totalBesoin + 1000 < this.peutproduireemploye * 1.0) ){
					if (contrat.getEcheancier().getStepFin()-contrat.getEcheancier().getStepDebut()<11
					|| contrat.getEcheancier().getStepDebut()-Filiere.LA_FILIERE.getEtape()>8) {
							return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, 1000 );
						} else { // les volumes sont corrects, la duree et le debut aussi
							return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, contrat.getEcheancier().getNbEcheances(), 1000 );
						}
				} else {
					return null;
				}
			}
			
		//si ce ne sont pas des Mirage; codé par Eliott
		} else {	
			
			Chocolat cD = ((ChocolatDeMarque)(contrat.getProduit())).getChocolat();
			
			if ((stockChoco.get( cD )-restantALivrerDeTypeAuStep(cD)-contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances()>5000)
			&& ( this.totalBesoin+contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances() < this.peutproduireemploye * 1.0)
			&& ( (contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances())<=2000 ) ){
				if (contrat.getEcheancier().getStepFin()-contrat.getEcheancier().getStepDebut()<11
					|| contrat.getEcheancier().getStepDebut()-Filiere.LA_FILIERE.getEtape()>8) {
					return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, contrat.getEcheancier().getQuantiteTotale()/12 );
				} else { // les volumes sont corrects, la duree et le debut aussi
					return contrat.getEcheancier();
				}
			} else if ((stockChoco.get( cD )-restantALivrerDeTypeAuStep(cD)-contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances()>5000)
					&& ( this.totalBesoin+contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances() < this.peutproduireemploye * 1.0)
					&& ( (contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances())>2000 ) ){
				if (contrat.getEcheancier().getStepFin()-contrat.getEcheancier().getStepDebut()<11
						|| contrat.getEcheancier().getStepDebut()-Filiere.LA_FILIERE.getEtape()>8) {
						return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, 2000 );
					} else { // les volumes sont corrects, la duree et le debut aussi
						return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, contrat.getEcheancier().getNbEcheances(), 2000 );
					}

			} else {
				if ((stockChoco.get(cD)-restantALivrerDeTypeAuStep(cD) - 1000> 5000)
				&& ( this.totalBesoin + 1000 < this.peutproduireemploye * 1.0) ){
					if (contrat.getEcheancier().getStepFin()-contrat.getEcheancier().getStepDebut()<11
					|| contrat.getEcheancier().getStepDebut()-Filiere.LA_FILIERE.getEtape()>8) {
						return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, 1000 );
					} else { // les volumes sont corrects, la duree et le debut aussi
						return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, contrat.getEcheancier().getNbEcheances(), 1000 );
					}
				} else {
					return null;
				}
			}
		}
	}

	public double getPrixFèves(IProduit p) {
		BourseCacao bourse = (BourseCacao)(Filiere.LA_FILIERE.getActeur("BourseCacao"));
		ChocolatDeMarque p2 = (ChocolatDeMarque)p ;
		Feve feve_utilise = Feve.F_HQ_BE; 
		double prix_F = 0.0 ;
		
		
		
		if (p2.getChocolat().equals(Chocolat.C_HQ_BE)) {
			feve_utilise = Feve.F_HQ_BE;  
		}
		if (p2.getChocolat().equals(Chocolat.C_HQ_E)) {
			feve_utilise = Feve.F_HQ_E;
		}
		if (p2.getChocolat().equals(Chocolat.C_HQ)) {
			feve_utilise = Feve.F_HQ;
		}
		if (p2.getChocolat().equals(Chocolat.C_MQ_E)) {
			feve_utilise = Feve.F_MQ_E;
		}
		if (p2.getChocolat().equals(Chocolat.C_MQ)) {
			feve_utilise = Feve.F_MQ;
		}
		if (p2.getChocolat().equals(Chocolat.C_BQ)) {
			feve_utilise = Feve.F_BQ;
		}
			
		double prixtot = 0.0; //prix a la tonne de tout les contrat concerant ce type de feve
		int i = 0; //nombre de contrat concerant ce tyoe de feve
		for (ExemplaireContratCadre c : this.contratsEnCours) { //on base le prix des fèves sur la moyenne pondérale des prix de contrat cadre par lesquels on les achète
			if (c.getProduit().equals(feve_utilise)) {
				i += 1;
				prixtot += c.getPrix();
			}
		}
		if (i == 0.0) { //si on a pas de contrat cadre (au début) pour les fèves, on se base sur le cours de la bourse pour des F_HQ
			if (feve_utilise.isEquitable() == true ) {
				prix_F = bourse.getCours(Feve.F_HQ).getValeur()*2;
			} else {
				prix_F = bourse.getCours(feve_utilise).getValeur()*1.5;
			}	
		} else {
			prix_F = prixtot/i;	
		}
		return prix_F;
	}
	
	
	public double propositionPrix(ExemplaireContratCadre contrat) {
		double prixPropose = 0.0;
		
		long N = contrat.getNumero();
		
		if (coutproduction_tonne_marque_step.isEmpty()){
			prixPropose = 1.1*(this.coutmachine + this.coutadjuvant*0.2 + getPrixFèves(contrat.getProduit()) + (1000*this.nbemployeCDI + 658)/(this.nbemployeCDI*this.tauxproductionemploye) );

		} else {
			prixPropose = 1.1*(coutproduction_tonne_marque_step.get(contrat.getProduit()) + getPrixFèves(contrat.getProduit()));

		}
		//System.out.println("prix tonne fève"+getPrixFèves(contrat.getProduit()));
		//System.out.println("prix propose est "+prixPropose);
		prixPrecedentC.put(	N, prixPropose);
		return prixPropose;
	}//prend compte des coûts de production en ajoutant une marge de 7%
	
	
	public double contrePropositionPrixVendeur(ExemplaireContratCadre contrat) {
		double prixPropose = contrat.getPrix();
		long N = contrat.getNumero();
		
		//double qte = contrat.getQuantiteTotale();
		
		///////début//////modif 07/05 Pierrick
		double coutProd = 0.0; //ce truc vaut la même chose que propositionPrix(contrat) mais sans la marge de x%
		if (coutproduction_tonne_marque_step.isEmpty()){
			coutProd = this.coutmachine + this.coutadjuvant*0.2 + getPrixFèves(contrat.getProduit()) + (1000*this.nbemployeCDI + 658)/(this.nbemployeCDI*this.tauxproductionemploye) ;
		} else {
			coutProd = coutproduction_tonne_marque_step.get(contrat.getProduit()) + getPrixFèves(contrat.getProduit());
		}
		//////fin/////////
		
		if (prixPrecedentC.get(N)==null) {
			if (prixPropose > propositionPrix(contrat)) {
				return prixPropose;
			} else {
				prixPrecedentC.put(N, propositionPrix(contrat));
				return propositionPrix(contrat);
			}
		} else {			
			if (prixPropose < prixPrecedentC.get(N)) {
				if ((prixPropose+prixPrecedentC.get(N))/2 >= coutProd*1.05){
					prixPrecedentC.replace(N, (prixPropose+prixPrecedentC.get(N))/2);
					return (prixPropose+prixPrecedentC.get(N))/2;
				} else if ((prixPropose+2*prixPrecedentC.get(N))/3 >= coutProd*1.05){
					prixPrecedentC.replace(N, (prixPropose+2*prixPrecedentC.get(N))/3);
					return (prixPropose+2*prixPrecedentC.get(N))/3;
				} else if ((prixPropose+3*prixPrecedentC.get(N))/4 >= coutProd*1.05){
					prixPrecedentC.replace(N, (prixPropose+3*prixPrecedentC.get(N))/4);
					return (prixPropose+3*prixPrecedentC.get(N))/4;
				} else {
					prixPrecedentC.replace(N, coutProd*1.05);
					return coutProd*1.05;
				}
			} else {
				return prixPropose;
			}
		}
	}//négocie le prix en nous garantissant une marge minimale de 5% (on enregistre notre précédente proposition dans prixPrecedent)

	
	
	
	
	//Après finalisation contrat 
	
	public double livrer(IProduit p, double quantite, ExemplaireContratCadre contrat) {
	
		//Si la livraison concerne des choco Mirage, codé par Anaïs et Pierrick
		if ( ((ChocolatDeMarque)p).getMarque() == "Mirage") {
				double stockMarque = (double)stockChocoMarque.get((ChocolatDeMarque)(contrat.getProduit()));
				
				if (quantite <= stockMarque) {
					journalVCC.ajouter("Livraison de "+quantite+" T de "+p+" du contrat "+contrat.getNumero());
					stockChocoMarque.put((ChocolatDeMarque)p, stockChocoMarque.get((ChocolatDeMarque)p)-quantite);
					totalStocksChocoMarque.retirer(this, quantite, cryptogramme);
					return quantite;
				}
				else {
					journalVCC.ajouter("Livraison de "+stockMarque+" T de "+p+" du contrat "+contrat.getNumero());
					stockChocoMarque.put((ChocolatDeMarque)p, stockChocoMarque.get((ChocolatDeMarque)p)-stockMarque);
					totalStocksChocoMarque.retirer(this, stockMarque, cryptogramme);
					return stockMarque;
				}
				
		//Si ce sont des chocos distributeurs, codé par Eliott
		} else {
			double stockDistrib = stockChoco.get( ((ChocolatDeMarque)(contrat.getProduit())).getChocolat() );
			
			if (quantite <= stockDistrib) {
				journalVCC.ajouter("Livraison de "+quantite+" T de "+p+" du contrat "+contrat.getNumero());
				stockChoco.put(((ChocolatDeMarque)p).getChocolat(), stockChoco.get(((ChocolatDeMarque)p).getChocolat())-quantite);
				totalStocksChoco.retirer(this, quantite, cryptogramme);
				return quantite;
			}
			else {
				journalVCC.ajouter("Livraison de "+stockDistrib+" T de "+p+" du contrat "+contrat.getNumero());
				stockChoco.put(((ChocolatDeMarque)p).getChocolat(), stockChoco.get(((ChocolatDeMarque)p).getChocolat())-stockDistrib);
				totalStocksChoco.retirer(this, stockDistrib, cryptogramme);
				return stockDistrib;
			}
		}
	}//s'inspirer de AcheteurCC
	
	public void notificationNouveauContratCadre(ExemplaireContratCadre contrat) {
		if (contrat.getVendeur().equals(this)) {
			journalVCC.ajouter("Nouveau contrat :"+contrat.getAcheteur() + " pour un total de " + contrat.getMontantRestantARegler() + " et un total de " + contrat.getQuantiteTotale() + " " + contrat.getProduit() + " s'étalant sur " + contrat.getEcheancier().getNbEcheances());
			this.contratsEnCours.add(contrat);
		}
		else {
			super.notificationNouveauContratCadre(contrat);
		}
	}

	
	
	
	
	//Honorer le contrat
	
	
	


	public double restantPayeARecevoir() {
		double res=0;
		for (ExemplaireContratCadre c : this.contratsEnCours) {
			res+=c.getMontantRestantARegler();
		}
		return res;
	}


	//Next
		
	public void next() { 
		super.next();
		this.journalVCC.ajouter("=== STEP "+Filiere.LA_FILIERE.getEtape()+" ====================");
		
		this.journalVCC.ajouter("===== phase de demande chocolat Mirage ======== ");
		
			//Pour les chocos de la marque CocOasis, codé par Anaïs et Pierrick

				for (ChocolatDeMarque choco : this.chocolatCocOasis) { // pas forcement equitable : on avise si on lance un contrat cadre pour tout type de feve
				if (vend(choco)) {

					if ((stockChocoMarque.get(choco)-this.restantALivrerAuStep(choco) > 5000) 
					&& ( this.totalBesoin < this.peutproduireemploye * 1.0)) {
						
						this.journalVCC.ajouter("restant a livrer de " + choco + " = " + this.restantALivrerAuStep(choco));
		
						
						
						this.journalVCC.ajouter("   "+choco+" suffisamment trop en stock/contrat pour passer un CC");
						double parStep = Math.max(1000, (-5000 + stockChocoMarque.get(choco) - this.restantALivrerAuStep(choco))); // au moins 100 par step
						if (parStep > 2000) {
							parStep = 2000;
						}
						Echeancier e = new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, parStep);	
						List<IAcheteurContratCadre> acheteurs = supCC.getAcheteurs(choco);
						if (acheteurs.size()>0) {
							for (IAcheteurContratCadre acheteur : acheteurs) {
								journalVCC.ajouter("   "+acheteur.getNom()+" retenu comme acheteur parmi "+acheteurs.size()+" acheteurs potentiels");
							
		
								ExemplaireContratCadre contrat = supCC.demandeVendeur(acheteur, this, choco, e, cryptogramme, true);
								if (contrat==null) {
									journalVCC.ajouter(Color.RED, Color.white,"   echec des negociations");
								} else {
									this.contratsEnCours.add(contrat);
									journalVCC.ajouter(Color.GREEN, Color.BLACK, "   contrat " + contrat.getNumero() + " signé avec l'équipe " + contrat.getAcheteur()+ " pour un total de " + contrat.getMontantRestantARegler()+ "euros et d'un stock de " + contrat.getQuantiteTotale() + " de " + contrat.getProduit());
								}
							}
							//IAcheteurContratCadre acheteur = acheteurs.get(Filiere.random.nextInt(acheteurs.size()));
							
							
						} else {
							journalVCC.ajouter("  pas d'acheteur");
						}
					} else {
						journalVCC.ajouter(" quantité de " + choco + "  insuffisante pour passer un contrat cadre");
					}
				}
				
				
				}
				
				journalVCC.ajouter(" === fin de phase de demande chocolat Mirage =========");
				this.journalVCC.ajouter("===== phase de demande chocolat Distributeur ======== ");
			//Pour les chocos de marque distributeur, codé par Eliott
				
				for(ChocolatDeMarque choco : this.chocolatDistributeur) {
				if (vend(choco)) {
						
					if ( (stockChoco.get(choco.getChocolat()) - this.restantALivrerDeTypeAuStep(choco.getChocolat())> 5000) 
					&& ( this.totalBesoin < this.peutproduireemploye * 1.0) ){ 
						
						this.journalVCC.ajouter("   "+choco+" suffisamment trop en stock/contrat pour passer un CC");
						double parStep = Math.max(1000, (-5000 + stockChoco.get(choco.getChocolat()) - this.restantALivrerDeTypeAuStep(choco.getChocolat()))); // au moins 100
						if (parStep > 2000) {
							parStep = 2000;
						}
						Echeancier e = new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, parStep);
						List<IAcheteurContratCadre> acheteurs = supCC.getAcheteurs(choco);
						
						if (acheteurs.size()>0) {

							for (IAcheteurContratCadre acheteur : acheteurs) {
									journalVCC.ajouter("   "+acheteur.getNom()+" retenu comme acheteur parmi "+acheteurs.size()+" acheteurs potentiels");

							/*IAcheteurContratCadre acheteur = acheteurs.get(Filiere.random.nextInt(acheteurs.size()));
							
							if (acheteur == acheteurPrecedent) {
								acheteurs.remove(acheteur);
								acheteur = acheteurs.get(Filiere.random.nextInt(acheteurs.size()));
							}
							
							acheteurPrecedent = acheteur;
							
							journalVCC.ajouter("   "+acheteur.getNom()+" retenu comme acheteur parmi "+acheteurs.size()+" acheteurs potentiels");
							*/
						
							
									ExemplaireContratCadre contrat = supCC.demandeVendeur(acheteur, this, choco, e, cryptogramme, true);
							
									if (contrat==null) {
										journalVCC.ajouter(Color.RED, Color.white,"   echec des negociations");
									} else {
										this.contratsEnCours.add(contrat);
										journalVCC.ajouter(Color.GREEN, Color.BLACK, "   contrat " + contrat.getNumero() + " signé avec l'équipe " + contrat.getAcheteur()+ " pour un total de " + contrat.getMontantRestantARegler() + "euros et d'un stock de " + contrat.getQuantiteTotale() + " de " + contrat.getProduit());
									}
								}
							//IAcheteurContratCadre acheteur = acheteurs.get(Filiere.random.nextInt(acheteurs.size()));
						} else {
							journalVCC.ajouter("   pas d'acheteur");
						}
					} else {
						journalVCC.ajouter(" quantité de " + choco + "  insuffisnate pour passer un contrat cadre");
					}	
				}
				
				}
				journalVCC.ajouter(" === fin de phase de demande chocolat Distributeur =========");
				
				journalVCC.ajouter("#######################################" + this.peutproduireemploye);
				journalVCC.ajouter("#######################################" + this.totalBesoin);
				
		// On archive les contrats terminés  (pas à modifier)
		for (ExemplaireContratCadre c : this.contratsEnCours) {
			if (c.getQuantiteRestantALivrer()==0.0 && c.getMontantRestantARegler()<=0.0) {
				this.contratsTermines.add(c);
			}
		}
		for (ExemplaireContratCadre c : this.contratsTermines) {
			if (c.getProduit().getType().equals("ChocolatDeMarque")) {
				journalVCC.ajouter("Archivage du contrat "+c);
			}
			
			this.contratsEnCours.remove(c);
		}
		this.journalVCC.ajouter("=====================================");
	}
	
	public List<Journal> getJournaux() {
		List<Journal> jx=super.getJournaux();
		jx.add(journalVCC);
		return jx;
	}

}
