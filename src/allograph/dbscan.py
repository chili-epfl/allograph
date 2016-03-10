import stroke
import learning_manager as lm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics


def main():
    """/////////////////////////////////////////////////////////INITIALIZATION//////////////////////////////////////////////////////////////////////////"""

    """ Create dictionnary : for a child -> strokes of the robot from the .dat files a the letter a"""
    strokes = {}
    strokes["Adele"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_adele", 0)['a']
    strokes["Alexandre"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_alexandre", 0)['a']
    strokes["Enzo"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_enzo", 0)['a']
    strokes["Jonathan"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_jonathan", 0)['a']
    strokes["Matenzo"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_matenzo", 0)['a']
    strokes["Mona"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_mona", 0)['a']
    strokes["Nathan"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_nathan", 0)['a']
    strokes["Valentine"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_valentine", 0)['a']
    strokes["Avery"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_avery", 0)['a']
    strokes["Dan"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_dan", 0)['a']
    strokes["DanielEge"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_daniel_ege", 0)['a']
    strokes["Gaia"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_gaia", 0)['a']
    strokes["Ines"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_ines", 0)['a']
    strokes["JacquelineNadine"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_jacqueline_nadine",
                 0)['a']
    strokes["Jake"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_jake", 0)['a']
    strokes["LaithKayra"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_laith_kayra", 0)['a']
    strokes["Lamonie"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_lamonie", 0)['a']
    strokes["LilaLudovica"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_lila_ludovica", 0)[
        'a']
    strokes["loulwaAnais"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_loulwa_anais", 0)[
        'a']
    strokes["Markus"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_markus", 0)['a']
    strokes["OsborneAmelia"] = lm.read_data(
        "/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_osborne_amelia_enzoV", 0)['a']
    strokes["Oscar"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_oscar", 0)['a']
    strokes["WilliamRin"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_william_rin", 0)['a']

    # ~ """Initialize Object KMeans that is goind to be used to find centroids on children's tries"""
    # ~ estimator = KMeans(init='k-means++')

    """////////////////////////////////////////////////////////////MANIPULATION////////////////////////////////////////////////////////////////////"""

    """Get the children's strokes from the robot's"""
    for key in strokes:
        strokes[key] = stroke.childFromRobot(strokes[key])
    """Build an array of all the strokes of all the children"""
    letters = []
    for key in strokes:
        print key
        for aStroke in strokes[key]:
            letters.append(stroke.strokeToArray(aStroke))
    print len(letters)

    letters = StandardScaler().fit_transform(letters)

    """Compute DBSCAN"""

    db = DBSCAN(eps=5.0, min_samples=4).fit(letters)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    """Number of clusters in labels, ignoring noise if present"""
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    # ~ print("Silhouette Coefficient: %0.3f"
    # ~ % metrics.silhouette_score(letters, labels))

    centroidStrokes = []

    """create strokes from centroids to be able to plot them easily"""
    clusters = [letters[labels == i] for i in xrange(n_clusters_)]
    print len(clusters)
    for cluster in clusters:
        # ~ for centroid in cluster:
        newStroke = stroke.Stroke()
        newStroke.stroke_from_xxyy(cluster[0])
        newStroke.downsampleShape(70)
        newStroke.uniformize()
        centroidStrokes.append(newStroke)

    """Plot the centroids"""
    # ~ centroidStrokes[0].plot()
    for aCentroid in centroidStrokes:
        aCentroid.plot()


if __name__ == '__main__':
    main()
