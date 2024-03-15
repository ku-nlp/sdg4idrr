PDTB3_L1_SENSES = [
    "Temporal",
    "Contingency",
    "Comparison",
    "Expansion",
]
PDTB3_L1_SENSE2DEFINITION = {
    "Temporal": "the situations described in the arguments are intended to be related temporally",
    "Contingency": "the situation described by one argument provides the reason, explanation or justification for the situation described by the other",
    "Comparison": "the discourse relation between two arguments highlights their differences or similarities, including differences between expected consequences and actual ones",
    "Expansion": "relations that expand the discourse and move its narrative or exposition forward",
}
SELECTED_PDTB3_L2_SENSES = [
    "Temporal.Synchronous",
    "Temporal.Asynchronous",
    "Contingency.Cause",
    "Contingency.Cause+Belief",
    "Contingency.Purpose",
    "Contingency.Condition",
    "Comparison.Concession",
    "Comparison.Contrast",
    "Expansion.Conjunction",
    "Expansion.Equivalence",
    "Expansion.Instantiation",
    "Expansion.Level-of-detail",
    "Expansion.Manner",
    "Expansion.Substitution",
]
SELECTED_PDTB3_L2_SENSE2DEFINITION = {
    "Temporal.Synchronous": "there is some degree of temporal overlap between the events described by the arguments",
    "Temporal.Asynchronous": "one event is described as preceding the other",
    "Contingency.Cause": "the situations described in the arguments are causally influenced but are not in a conditional relation",
    "Contingency.Cause+Belief": "evidence is provided to cause the hearer to believe a claim",
    "Contingency.Purpose": "one argument presents an action that an agent undertakes with the purpose of the goal conveyed by the other argument being achieved",
    "Contingency.Condition": "one argument presents a situation as unrealized (the antecedent), which (when realized) would lead to the situation described by the other argument",
    "Comparison.Concession": "an expected causal relation is cancelled or denied by the situation described in one of the arguments",
    "Comparison.Contrast": "at least two differences between the arguments are highlighted",
    "Expansion.Conjunction": "both arguments, which don’t directly relate to each other, bear the same relation to some other situation evoked in the discourse",
    "Expansion.Equivalence": "both arguments are taken to describe the same situation, but from different perspectives",
    "Expansion.Instantiation": "one argument describes a situation as holding in a set of circumstances, while the other argument describes one or more of those circumstances",
    "Expansion.Level-of-detail": "both arguments describe the same situation, but in less or more detail",
    "Expansion.Manner": "the situation described by one argument presents the manner in which the situation described by other argument has happened or been done",
    "Expansion.Substitution": "arguments are presented as exclusive alternatives, with one being ruled out",
}
