package abstraction.eq7Transformateur4;

import java.util.List;

import abstraction.eqXRomu.bourseCacao.IAcheteurBourse;
import abstraction.eqXRomu.contratsCadres.ExemplaireContratCadre;
import abstraction.eqXRomu.filiere.Filiere;
import abstraction.eqXRomu.general.Journal;
import abstraction.eqXRomu.produits.Chocolat;
import abstraction.eqXRomu.produits.ChocolatDeMarque;
import abstraction.eqXRomu.produits.Feve;



public class Transformateur4AcheteurBourse extends Transformateur4Acteur implements IAcheteurBourse {

	protected Journal journalBourse;
	private double D;
	
	public Transformateur4AcheteurBourse () {
		super();
		this.journalBourse = new Journal(this.getNom()+" journal Bourse achat", this);
	}
	
	


	public double demande(Feve f, double cours) { //changer selon conditions et qte d'achat de chaque fève
		if (f.equals(Feve.F_BQ) && stockFeves.get(f) + getQuantiteAuStep(f) - BesoinDeFeve(f) < 10000 ) {
			D = 10000 - (stockFeves.get(f) + getQuantiteAuStep(f) - BesoinDeFeve(f)) ;
			journalBourse.ajouter(Filiere.LA_FILIERE.getEtape()+" : je souhaite acheter "+ D +" T de "+f);
			return D;
		}
		//if (f.equals(Feve.F_HQ) && stockFeves.get(f) + getQuantiteAuStep(f) - BesoinDeFeve(f) < 10000 ) {
		//	D = 10000 - (stockFeves.get(f) + getQuantiteAuStep(f) - BesoinDeFeve(f)) ;
		//	journalBourse.ajouter(Filiere.LA_FILIERE.getEtape()+" : je souhaite acheter "+ D +" T de "+f);
		//	return D;
		//}
		else {
			return 0;
		}
	}
	
	
	
	public void notificationAchat(Feve f, double quantiteEnT, double coursEnEuroParT) {
		this.stockFeves.put(f, this.stockFeves.get(f)+quantiteEnT);
		this.totalStocksFeves.ajouter(this, quantiteEnT, cryptogramme);
		this.journalBourse.ajouter("- achat de "+quantiteEnT+"T de fèves "+f);
	}

	public void notificationBlackList(int dureeEnStep) {
		this.journalBourse.ajouter(Filiere.LA_FILIERE.getEtape()+"blacklisté pendant"+dureeEnStep+"etapes");
	}
	
	
	public List<Journal> getJournaux() {
		List<Journal> res=super.getJournaux();
		res.add(journalBourse);
		return res;
	}
	
	public void next() {
		super.next();
		this.journalBourse.ajouter("=== STEP "+Filiere.LA_FILIERE.getEtape()+" ====================");
	}
}
