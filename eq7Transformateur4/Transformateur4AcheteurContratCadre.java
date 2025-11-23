package abstraction.eq7Transformateur4;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import abstraction.eqXRomu.bourseCacao.BourseCacao;
import abstraction.eqXRomu.contratsCadres.Echeancier;
import abstraction.eqXRomu.contratsCadres.ExemplaireContratCadre;
import abstraction.eqXRomu.contratsCadres.IAcheteurContratCadre;
import abstraction.eqXRomu.contratsCadres.IVendeurContratCadre;
import abstraction.eqXRomu.contratsCadres.SuperviseurVentesContratCadre;
import abstraction.eqXRomu.produits.Chocolat;
import abstraction.eqXRomu.produits.ChocolatDeMarque;
import abstraction.eqXRomu.produits.Feve;
import abstraction.eqXRomu.produits.IProduit;
import abstraction.eqXRomu.filiere.*;
import abstraction.eqXRomu.general.Journal;
import java.awt.Color;
import java.awt.Color;

//codé par Anaïs

public class Transformateur4AcheteurContratCadre extends Transformation2 implements IAcheteurContratCadre{
	
	private SuperviseurVentesContratCadre supCC;
	
	protected List<ExemplaireContratCadre> contratsTermines;
	protected Journal journalACC;
	private HashMap<Long, Double> prixPrecedentF;

	public Transformateur4AcheteurContratCadre() {
		super();
		
		this.contratsTermines=new LinkedList<ExemplaireContratCadre>();
		this.journalACC = new Journal(this.getNom()+" journal CC achat", this);
		this.prixPrecedentF = new HashMap< Long ,Double>();

	}
	
	public void initialiser() {
		super.initialiser();
		this.supCC = (SuperviseurVentesContratCadre)(Filiere.LA_FILIERE.getActeur("Sup.CCadre"));
		
	}
	
	public boolean achete(IProduit produit) { //on n'achête que des feves HQ_BE, HQ ou MQ
		return (produit.getType().equals("Feve"))
				&& (

						//(((Feve)produit).equals(Feve.F_HQ)) ||
						(((Feve)produit).equals(Feve.F_HQ_E)) ||
						(((Feve)produit).equals(Feve.F_BQ))
					)
				
				&& (stockFeves.get(produit)+restantDu((Feve)produit) < 20000 );
	}
	
	
	
	
	
	
	//Négociations
	
	public Echeancier contrePropositionDeLAcheteur(ExemplaireContratCadre contrat) {   
		
		
		double BesoinPourChoco = 0.0;
		
		//à modifier selon comment on veut nos échéanciers
		if (!(contrat.getProduit().getType().equals("Feve"))) {
			return null;
		}
		
		Feve f = (Feve)(contrat.getProduit()) ;
		
		BesoinPourChoco = BesoinDeFeve(f);
				
		//for (ExemplaireContratCadre c : this.contratsEnCours) {
		//	if ( (c.getProduit().getType().equals("ChocolatDeMarque")) 
		//		&& (((ChocolatDeMarque)(c.getProduit())).getChocolat().getGamme().equals(((Feve)(contrat.getProduit())).getGamme() ) )) {
		//		
		//	}
		//}

		if ((stockFeves.get(f)+this.getQuantiteAuStep(f)+contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances() - BesoinPourChoco < 10000)
				&& (contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances() <= 4000)){
			if (contrat.getEcheancier().getStepFin()-contrat.getEcheancier().getStepDebut()<11
			|| contrat.getEcheancier().getStepDebut()-Filiere.LA_FILIERE.getEtape()>8) { //contrat de minimum 12 steps
				if (contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances() < 100) { //minimum 1000 T par steps
					return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, 100 );
				} else {
					return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, contrat.getEcheancier().getQuantiteTotale()/12 );
				}
			} else { // les volumes peuvent être acceptable, la duree et le debut aussi
				if (contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances() < 100) { //minimum 1000 T par steps
					return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, contrat.getEcheancier().getNbEcheances(), 100 );
				} else {
					return contrat.getEcheancier();
				}
			}
		} else if ((stockFeves.get(f)+this.getQuantiteAuStep(f)+contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances() - BesoinPourChoco < 10000)
				&& (contrat.getEcheancier().getQuantiteTotale()/contrat.getEcheancier().getNbEcheances() > 4000)) {
			if (contrat.getEcheancier().getStepFin()-contrat.getEcheancier().getStepDebut()<11
					|| contrat.getEcheancier().getStepDebut()-Filiere.LA_FILIERE.getEtape()>8) { //contrat de minimum 12 steps
							return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, 4000 );
					} else { // les volumes peuvent être acceptable, la duree et le debut aussi
							return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, contrat.getEcheancier().getNbEcheances(), 4000 );
					}

		} else { //nous ne sommes pas en besoin de feve 
			return null;
			//double marge = 15000 - stockFeves.get((Feve)(contrat.getProduit())) - restantDu((Feve)(contrat.getProduit()));
			//if (marge<1200) {
			//	return null;
			//} else {
			//	double quantite = 1200 + Filiere.random.nextDouble()*(marge-1200); // un nombre aleatoire entre 1200 et la marge
			//	return new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, quantite/12 );
			//}
		}
	}

	public double contrePropositionPrixAcheteur(ExemplaireContratCadre contrat) {  //à modifier selon ce qu'on est prêt à payer pour quoi
		// Il faudrait normalement tenir compte du volume du contrat (plus le volume est important 
		// plus les prix seront bas) et de l'urgence (si on n'en n'a pas en stock et pas de CC alors 
		// il devient plus urgent d'en disposer et donc on acceptera davantage un prix eleve)
		// mais dans cet acteur trivial on ne se base que sur le prix et via des tirages aleatoires.
		double prixPropose = contrat.getPrix();
		long N = contrat.getNumero();
		
		BourseCacao bourse = (BourseCacao)(Filiere.LA_FILIERE.getActeur("BourseCacao"));
		double solde = Filiere.LA_FILIERE.getBanque().getSolde(this, cryptogramme)-restantAPayer();
		double prixSansDecouvert = solde / contrat.getQuantiteTotale();
		if (prixSansDecouvert<bourse.getCours(Feve.F_BQ).getValeur()) {
			return 0.0; // nous ne sommes pas en mesure de fournir un prix raisonnable
		}
		double prix = 0.0 ;
		Feve feve_utilise = (Feve)contrat.getProduit();
		
		
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
				prix = bourse.getCours(Feve.F_HQ).getValeur()*2;//testP *2
			} else {
				prix = bourse.getCours(feve_utilise).getValeur()*1.75;//testP *1.5
			}	
		} else {
			prix = prixtot/i;	
		}
	
		if (prixPrecedentF.get(N)==null) {	
			if (prixPropose < prix) {
				return prixPropose;
			} else {
				prixPrecedentF.put(N,prix);
				return prix;
			}
		} else {
			if (prixPropose > prixPrecedentF.get(N)) {
				if ((prixPropose+prixPrecedentF.get(N))/2 <= prix*1.3){// *1.2
					prixPrecedentF.replace(N,(prixPropose+prixPrecedentF.get(N))/2);
					return prixPrecedentF.get(N);
				} else if ((prixPropose+2*prixPrecedentF.get(N))/3 <= prix*1.3){ // *1.2
					prixPrecedentF.replace(N,(prixPropose+2*prixPrecedentF.get(N))/3);
					return prixPrecedentF.get(N);
				} else if ((prixPropose+3*prixPrecedentF.get(N))/4 <= prix*1.3){ // *1.2
					prixPrecedentF.replace(N,(prixPropose+3*prixPrecedentF.get(N))/4);
					return prixPrecedentF.get(N);
				} else {
					prixPrecedentF.replace(N, prix*1.3); // *1.2
					return prixPrecedentF.get(N);
				}
			} else {
				return prixPropose;
			}
		}
	}
		
	
	
	//Après finalisation contrat 
	
	public void notificationNouveauContratCadre(ExemplaireContratCadre contrat) {
		journalACC.ajouter("Nouveau contrat :"+contrat);
		this.contratsEnCours.add(contrat);
	}

	public void receptionner(IProduit p, double quantiteEnTonnes, ExemplaireContratCadre contrat) {
		journalACC.ajouter("Reception de "+quantiteEnTonnes+" T de "+p+" du contrat "+contrat.getNumero());
		stockFeves.put((Feve)p, stockFeves.get((Feve)p)+quantiteEnTonnes);
		totalStocksFeves.ajouter(this, quantiteEnTonnes, cryptogramme);		
	}
	
	
	
	
	
	
	//Honorer les contrats


	
	public double restantAPayer() {
		double res=0;
		for (ExemplaireContratCadre c : this.contratsEnCours) {
			res+=c.getMontantRestantARegler();
		}
		return res;
	}
	
	
	
	//Next
	
	public void next() { 	//à modifier selon nb de fèves qu'on veut
		super.next();
		this.journalACC.ajouter("=== STEP "+Filiere.LA_FILIERE.getEtape()+" ====================");
		
		
		List<Feve> fInteresse = new LinkedList<Feve>();
		//fInteresse.add(Feve.F_HQ_BE);

		//fInteresse.add(Feve.F_HQ);
		fInteresse.add(Feve.F_BQ);
		fInteresse.add(Feve.F_HQ_E);
				for (Feve f : fInteresse) { // pas forcement equitable : on avise si on lance un contrat cadre pour tout type de feve
					
					double BesoinPourChoco = 0.0; //quantité à livrer de fèves
					
					BesoinPourChoco = BesoinDeFeve(f);

					
					if (stockFeves.get(f)+this.getQuantiteAuStep(f) - BesoinPourChoco < 10000 ) { 
						this.journalACC.ajouter("   "+f+" suffisamment peu en stock/contrat pour passer un CC");
						double parStep = Math.max(500, (1000-this.getQuantiteAuStep(f)+BesoinPourChoco)); // au moins 500
						Echeancier e = new Echeancier(Filiere.LA_FILIERE.getEtape()+1, 12, parStep);
						List<IVendeurContratCadre> vendeurs = supCC.getVendeurs(f);
						if (vendeurs.size()>0) {
							for (IVendeurContratCadre vendeur : vendeurs) {
								journalACC.ajouter("   "+vendeur.getNom()+" retenu comme vendeur parmi "+vendeurs.size()+" vendeurs potentiels");
								ExemplaireContratCadre contrat = supCC.demandeAcheteur(this, vendeur, f, e, cryptogramme, false);
								if (contrat==null) {
									journalACC.ajouter(Color.RED, Color.white,"   echec des negociations");
								} else {
									this.contratsEnCours.add(contrat);
									journalACC.ajouter(Color.GREEN, vendeur.getColor(), "   contrat signe");
								}
							}
							//IVendeurContratCadre vendeur = vendeurs.get(Filiere.random.nextInt(vendeurs.size()));
							
						} else {
							journalACC.ajouter("   pas de vendeur");
						}
					}
				}
	
					
					
				
	
		// On archive les contrats terminés  (pas à modifier)
		for (ExemplaireContratCadre c : this.contratsEnCours) {
			if (c.getQuantiteRestantALivrer()==0.0 && c.getMontantRestantARegler()<=0.0) {
				this.contratsTermines.add(c);
			}
		}
		for (ExemplaireContratCadre c : this.contratsTermines) {
			if (c.getProduit().getType().equals("Feve")) {
				journalACC.ajouter("Archivage du contrat "+c);
			}
			this.contratsEnCours.remove(c);
		}
		this.journalACC.ajouter("=================================");
	}
	
	public List<Journal> getJournaux() {
		List<Journal> jx=super.getJournaux();
		jx.add(journalACC);
		return jx;
	}
}
