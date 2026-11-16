//! Canonical humanoid skeleton and bone-name canonicalisation.
//!
//! Vendored from `@three-ws/retarget` 0.1.2 (Apache-2.0). The bone set, its
//! rest quaternions, the hierarchy and the alias table are the package's own
//! data (`src/glb-canonicalize.js`, `src/animation-canonical-rest.js`), copied
//! verbatim so a rig resolves to the same bone here as it does there. Only the
//! lookup code was rewritten; the tables are the contract.
//!
//! The package ships **52** canonical bones (its README says 53, its data says
//! 52 — `Hips` … `RightToeBase`, Mixamo naming, fingers included) and **715**
//! aliases. Vendors whose rigs are covered by the alias table: Mixamo,
//! Avaturn/Wolf3D, VRM 0.x (`J_Bip_*`) and 1.0, Daz/Genesis, Reallusion CC3/CC4,
//! MakeHuman, Unreal, HumanIK/Maya, Blender/Rigify `.L`/`.R`, Roblox R15/R6,
//! Second Life, Advanced Skeleton, MMD/PMX (Japanese names), anatomically named
//! rigs (`humerus.L`, `femur.L`), plus casing and separator variants.

/// Canonical bones, in the package's order.
pub const BONE_COUNT: usize = 52;

/// Below this share of matched bones a rig is not a humanoid, and a retarget
/// must refuse rather than produce half a puppet.
pub const MIN_COVERAGE: f32 = 0.5;

pub const BONES: [&str; BONE_COUNT] = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
];

/// Parent of each bone, as an index into [`BONES`]; `None` at the root.
pub const PARENTS: [Option<u8>; BONE_COUNT] = [
    None,
    Some(0),
    Some(1),
    Some(2),
    Some(3),
    Some(4),
    Some(3),
    Some(6),
    Some(7),
    Some(8),
    Some(9),
    Some(10),
    Some(11),
    Some(9),
    Some(13),
    Some(14),
    Some(9),
    Some(16),
    Some(17),
    Some(9),
    Some(19),
    Some(20),
    Some(9),
    Some(22),
    Some(23),
    Some(3),
    Some(25),
    Some(26),
    Some(27),
    Some(28),
    Some(29),
    Some(30),
    Some(28),
    Some(32),
    Some(33),
    Some(28),
    Some(35),
    Some(36),
    Some(28),
    Some(38),
    Some(39),
    Some(28),
    Some(41),
    Some(42),
    Some(0),
    Some(44),
    Some(45),
    Some(46),
    Some(0),
    Some(48),
    Some(49),
    Some(50),
];

/// Rest orientation per bone, `[x, y, z, w]`, in the package's canonical space.
pub const REST: [[f32; 4]; BONE_COUNT] = [
    [0.0010741148761599644, 6537898515070053e-23, 7022659074474357e-26, 0.9999994231384476],
    [-0.018332562694809717, -14544833124647894e-23, 26976261645093277e-26, 0.9998319444511763],
    [-0.03815628571181615, 20010687323521798e-24, 534807122500512e-23, 0.9992717837809081],
    [-0.056998031890724846, 13546308636882866e-23, -8861325315468363e-25, 0.9983742907149431],
    [0.2225107795455935, -23352293238692193e-23, -19647139012020554e-24, 0.974930229804142],
    [-0.11706881878098431, 9438845082943482e-24, 372654613409334e-22, 0.993123804804428],
    [0.5181706903502131, 0.5134387363832796, -0.48299308003518876, 0.4843526489955184],
    [-0.003089573263103852, 0.08018148459223169, -0.025041829737156785, 0.9964608827384491],
    [-26826767953574183e-23, 12816740072681333e-23, 0.012369379808617674, 0.9999234962951223],
    [-0.0012910117424963744, 0.002537815819541244, 0.010116448936771993, 0.9999447735952517],
    [0.028803290204768833, 0.011362422123976328, 0.055965724298468224, 0.9979524555518633],
    [-0.005195237841915152, 0.014315447122751807, 21581682553071412e-20, 0.9998840087229335],
    [-0.0013151015878908295, -0.016941197343945148, -0.006458334440614796, 0.9998347644768737],
    [0.027532617383276645, -0.031607746403852485, 0.016931179482824785, 0.9989775976007271],
    [-0.001267981765488788, 0.0010468621884392817, -0.004466175725402764, 0.9999886747239598],
    [-0.0015803533040544947, 0.0018352886876119797, 0.0038040706030556736, 0.9999898315711762],
    [0.009345046842530319, -0.07377447953394943, -0.04207784173776388, 0.9963430390695086],
    [-0.0019002180814112143, 2770864318134059e-19, 4968185265866393e-19, 0.9999980327809131],
    [-0.0014157518824255099, -0.005595319155644295, -0.004606740515194791, 0.9999727326242506],
    [0.018207541076892743, -0.03436008716699126, -0.036855349348364555, 0.9985637451229231],
    [-6571286469542868e-19, 0.015477225242721515, 0.01003471962236163, 0.9998296495318929],
    [0.001815607007583071, -0.003099595768579846, 0.0072046170631637615, 0.999967594260012],
    [0.1900227602341495, -0.058826116539083484, 0.3478783626049766, 0.9161940209594684],
    [0.01801363066187901, -0.020968588375318502, -0.22063576333056306, 0.9749644544258214],
    [-0.014498565873099356, 0.03466160908487025, 0.011599642488048602, 0.9992266072970502],
    [0.5181706006117709, -0.5134387953473804, 0.4829931766210871, 0.4843525890882128],
    [-0.003089570938390283, -0.08018164882954634, 0.025041835809991725, 0.9964608693782484],
    [-2706717415294191e-22, 60675494372234465e-25, -0.012369382961452784, 0.9999234962561284],
    [-0.0012910114923260908, -0.002537815924782367, -0.010116450486313036, 0.9999447735796301],
    [0.028803284769926347, -0.01136242475457741, -0.05596571258489258, 0.9979524563349205],
    [-0.005195229421057241, -0.014315442233259087, -21582506177354608e-20, 0.9998840088349297],
    [-0.0013151045827228072, 0.016941319655899355, 0.006458327385746035, 0.9998347624459482],
    [0.027532662650122483, 0.03160778647114273, -0.01693120529165708, 0.9989775946478099],
    [-0.0012680273836480952, -0.0010469047250576836, 0.004466199390698661, 0.9999886745158897],
    [-0.0015803562094451136, -0.001835288773706009, -0.003804068167924464, 0.9999898315756854],
    [0.009345030274151258, 0.07377446325595405, 0.04207784564870347, 0.9963430402658229],
    [-0.0019002008217559384, -2769620503994165e-19, -4968229018870381e-19, 0.9999980328459891],
    [-0.0014157497333212687, 0.0055951881780210025, 0.004606743315319843, 0.9999727333471813],
    [0.01820754441109714, 0.03436008722907556, 0.03685535429214536, 0.9985637448787448],
    [-6571335802527884e-19, -0.015477230302375519, -0.010034723368684675, 0.999829649412505],
    [0.0018156069598940002, 0.003099598646436032, -0.007204626788931167, 0.9999675941811231],
    [0.19002276085311107, 0.05882618704962336, -0.3478783387552461, 0.9161940229953188],
    [0.018013663274561078, 0.020968550782212388, 0.22063584956274163, 0.9749644350943654],
    [-0.014498582331377526, -0.03466161961137741, -0.011599713293476643, 0.9992266058762234],
    [-4952055077835167e-20, 0.0032567616062252223, 0.9999946186652781, -39203576089550014e-20],
    [-0.022987686472704008, -6608912898930547e-19, 7960712093047862e-20, 0.9997355266049326],
    [0.5523737034239063, 2518875693487157e-19, 0.0036538113016644914, 0.8335885543726441],
    [0.23652217735689587, 6644108661216591e-19, -0.002051152932444189, 0.9716236982227819],
    [49438593286575395e-21, 0.003256759408127677, 0.9999946186765544, 3920356017771739e-19],
    [-0.022987685898989207, 6609402907891899e-19, -7960698692017099e-20, 0.9997355265857409],
    [0.5523737112340288, -25188760500252676e-20, -0.0036538107149212758, 0.8335885491999105],
    [0.23652222014073046, -6644109035597298e-19, 0.0020511526077556528, 0.9716236878087238],
];

/// Lower-cased, separator-free alias -> bone index.
///
/// The package's full map: its canonical names, its Unreal table and the extras
/// (VRM 0.x/1.0, MMD/PMX, DAZ/Reallusion, Second Life, Advanced Skeleton,
/// Blender/Rigify `.L`/`.R`, anatomical spellings …), generated with the
/// package's own first-wins precedence and dumped through its
/// `canonicalizeBoneName`. Sorted by key; a linear scan is fine because a rig is
/// resolved once, not per frame.
pub const ALIASES: &[(&str, u8)] = &[
    ("abdomen", 1),
    ("abdomenlower", 1),
    ("abdomenupper", 2),
    ("anklel", 46),
    ("ankler", 50),
    ("balll", 47),
    ("ballr", 51),
    ("calcaneusl", 46),
    ("calcaneusr", 50),
    ("calfl", 45),
    ("calfr", 49),
    ("carpusl", 9),
    ("carpusr", 28),
    ("cervical", 4),
    ("chest", 2),
    ("chestlower", 2),
    ("chestm", 3),
    ("chestupper", 3),
    ("claviclel", 6),
    ("clavicler", 25),
    ("collarl", 6),
    ("collarr", 25),
    ("cranium", 5),
    ("elbowl", 8),
    ("elbowr", 27),
    ("femurl", 44),
    ("femurr", 48),
    ("fibulal", 45),
    ("fibular", 49),
    ("findex01l", 10),
    ("findex01r", 29),
    ("findex02l", 11),
    ("findex02r", 30),
    ("findex03l", 12),
    ("findex03r", 31),
    ("finger11l", 22),
    ("finger11r", 41),
    ("finger12l", 23),
    ("finger12r", 42),
    ("finger13l", 24),
    ("finger13r", 43),
    ("finger21l", 10),
    ("finger21r", 29),
    ("finger22l", 11),
    ("finger22r", 30),
    ("finger23l", 12),
    ("finger23r", 31),
    ("finger31l", 13),
    ("finger31r", 32),
    ("finger32l", 14),
    ("finger32r", 33),
    ("finger33l", 15),
    ("finger33r", 34),
    ("finger41l", 19),
    ("finger41r", 38),
    ("finger42l", 20),
    ("finger42r", 39),
    ("finger43l", 21),
    ("finger43r", 40),
    ("finger51l", 16),
    ("finger51r", 35),
    ("finger52l", 17),
    ("finger52r", 36),
    ("finger53l", 18),
    ("finger53r", 37),
    ("flittle01l", 16),
    ("flittle01r", 35),
    ("flittle02l", 17),
    ("flittle02r", 36),
    ("flittle03l", 18),
    ("flittle03r", 37),
    ("fmiddle01l", 13),
    ("fmiddle01r", 32),
    ("fmiddle02l", 14),
    ("fmiddle02r", 33),
    ("fmiddle03l", 15),
    ("fmiddle03r", 34),
    ("footl", 46),
    ("footr", 50),
    ("forearml", 8),
    ("forearmr", 27),
    ("fpinky01l", 16),
    ("fpinky01r", 35),
    ("fpinky02l", 17),
    ("fpinky02r", 36),
    ("fpinky03l", 18),
    ("fpinky03r", 37),
    ("fring01l", 19),
    ("fring01r", 38),
    ("fring02l", 20),
    ("fring02r", 39),
    ("fring03l", 21),
    ("fring03r", 40),
    ("fthumb01l", 22),
    ("fthumb01r", 41),
    ("fthumb02l", 23),
    ("fthumb02r", 42),
    ("fthumb03l", 24),
    ("fthumb03r", 43),
    ("handl", 9),
    ("handr", 28),
    ("head", 5),
    ("headm", 5),
    ("hip", 0),
    ("hipl", 44),
    ("hipm", 0),
    ("hipmaster", 0),
    ("hipr", 48),
    ("hips", 0),
    ("humerusl", 7),
    ("humerusr", 26),
    ("index01l", 10),
    ("index01r", 29),
    ("index02l", 11),
    ("index02r", 30),
    ("index03l", 12),
    ("index03r", 31),
    ("index1l", 10),
    ("index1r", 29),
    ("index2l", 11),
    ("index2r", 30),
    ("index3l", 12),
    ("index3r", 31),
    ("indexdistall", 12),
    ("indexdistalr", 31),
    ("indexfinger1l", 10),
    ("indexfinger1r", 29),
    ("indexfinger2l", 11),
    ("indexfinger2r", 30),
    ("indexfinger3l", 12),
    ("indexfinger3r", 31),
    ("indexfingerl", 10),
    ("indexfingerr", 29),
    ("indexintermediatel", 11),
    ("indexintermediater", 30),
    ("indexl", 10),
    ("indexproximall", 10),
    ("indexproximalr", 29),
    ("indexr", 29),
    ("jbipcchest", 2),
    ("jbipchead", 5),
    ("jbipchips", 0),
    ("jbipcneck", 4),
    ("jbipcspine", 1),
    ("jbipcupperchest", 3),
    ("jbiplfoot", 46),
    ("jbiplhand", 9),
    ("jbiplindex1", 10),
    ("jbiplindex2", 11),
    ("jbiplindex3", 12),
    ("jbipllittle1", 16),
    ("jbipllittle2", 17),
    ("jbipllittle3", 18),
    ("jbipllowerarm", 8),
    ("jbipllowerleg", 45),
    ("jbiplmiddle1", 13),
    ("jbiplmiddle2", 14),
    ("jbiplmiddle3", 15),
    ("jbiplring1", 19),
    ("jbiplring2", 20),
    ("jbiplring3", 21),
    ("jbiplshoulder", 6),
    ("jbiplthumb1", 22),
    ("jbiplthumb2", 23),
    ("jbiplthumb3", 24),
    ("jbipltoebase", 47),
    ("jbipltoes", 47),
    ("jbiplupperarm", 7),
    ("jbiplupperleg", 44),
    ("jbiprfoot", 50),
    ("jbiprhand", 28),
    ("jbiprindex1", 29),
    ("jbiprindex2", 30),
    ("jbiprindex3", 31),
    ("jbiprlittle1", 35),
    ("jbiprlittle2", 36),
    ("jbiprlittle3", 37),
    ("jbiprlowerarm", 27),
    ("jbiprlowerleg", 49),
    ("jbiprmiddle1", 32),
    ("jbiprmiddle2", 33),
    ("jbiprmiddle3", 34),
    ("jbiprring1", 38),
    ("jbiprring2", 39),
    ("jbiprring3", 40),
    ("jbiprshoulder", 25),
    ("jbiprthumb1", 41),
    ("jbiprthumb2", 42),
    ("jbiprthumb3", 43),
    ("jbiprtoebase", 51),
    ("jbiprtoes", 51),
    ("jbiprupperarm", 26),
    ("jbiprupperleg", 48),
    ("kneel", 45),
    ("kneer", 49),
    ("lankle", 46),
    ("larm", 7),
    ("lcalcaneus", 46),
    ("lcalf", 45),
    ("lcarpus", 9),
    ("lclavicle", 6),
    ("lcollar", 6),
    ("leftankle", 46),
    ("leftarm", 7),
    ("leftcalf", 45),
    ("leftclavicle", 6),
    ("leftcollar", 6),
    ("leftelbow", 8),
    ("leftfoot", 46),
    ("leftforearm", 8),
    ("lefthand", 9),
    ("lefthandindex1", 10),
    ("lefthandindex2", 11),
    ("lefthandindex3", 12),
    ("lefthandmiddle1", 13),
    ("lefthandmiddle2", 14),
    ("lefthandmiddle3", 15),
    ("lefthandpinky1", 16),
    ("lefthandpinky2", 17),
    ("lefthandpinky3", 18),
    ("lefthandring1", 19),
    ("lefthandring2", 20),
    ("lefthandring3", 21),
    ("lefthandthumb1", 22),
    ("lefthandthumb2", 23),
    ("lefthandthumb3", 24),
    ("lefthip", 44),
    ("leftindexdistal", 12),
    ("leftindexintermediate", 11),
    ("leftindexproximal", 10),
    ("leftknee", 45),
    ("leftleg", 45),
    ("leftlittledistal", 18),
    ("leftlittleintermediate", 17),
    ("leftlittleproximal", 16),
    ("leftlowerarm", 8),
    ("leftlowerleg", 45),
    ("leftmiddledistal", 15),
    ("leftmiddleintermediate", 14),
    ("leftmiddleproximal", 13),
    ("leftpinkydistal", 18),
    ("leftpinkyintermediate", 17),
    ("leftpinkyproximal", 16),
    ("leftringdistal", 21),
    ("leftringintermediate", 20),
    ("leftringproximal", 19),
    ("leftshin", 45),
    ("leftshoulder", 6),
    ("leftthigh", 44),
    ("leftthumbdistal", 24),
    ("leftthumbmetacarpal", 22),
    ("leftthumbproximal", 23),
    ("lefttoe", 47),
    ("lefttoebase", 47),
    ("lefttoes", 47),
    ("leftupleg", 44),
    ("leftupperarm", 7),
    ("leftupperleg", 44),
    ("leftwrist", 9),
    ("lelbow", 8),
    ("lfemur", 44),
    ("lfibula", 45),
    ("lfoot", 46),
    ("lforearm", 8),
    ("lforearmbend", 8),
    ("lhand", 9),
    ("lhip", 44),
    ("lhumerus", 7),
    ("lindex1", 10),
    ("lindex2", 11),
    ("lindex3", 12),
    ("little01l", 16),
    ("little01r", 35),
    ("little02l", 17),
    ("little02r", 36),
    ("little03l", 18),
    ("little03r", 37),
    ("little1l", 16),
    ("little1r", 35),
    ("little2l", 17),
    ("little2r", 36),
    ("little3l", 18),
    ("little3r", 37),
    ("littledistall", 18),
    ("littledistalr", 37),
    ("littlefinger1l", 16),
    ("littlefinger1r", 35),
    ("littlefinger2l", 17),
    ("littlefinger2r", 36),
    ("littlefinger3l", 18),
    ("littlefinger3r", 37),
    ("littlefingerl", 16),
    ("littlefingerr", 35),
    ("littleintermediatel", 17),
    ("littleintermediater", 36),
    ("littlel", 16),
    ("littleproximall", 16),
    ("littleproximalr", 35),
    ("littler", 35),
    ("lknee", 45),
    ("lleg", 45),
    ("llittle1", 16),
    ("llittle2", 17),
    ("llittle3", 18),
    ("llowerarm", 8),
    ("llowerleg", 45),
    ("lmetatarsus", 47),
    ("lmid1", 13),
    ("lmid2", 14),
    ("lmid3", 15),
    ("lmiddle1", 13),
    ("lmiddle2", 14),
    ("lmiddle3", 15),
    ("lowerarml", 8),
    ("lowerarmr", 27),
    ("lowerchest", 2),
    ("lowerlegl", 45),
    ("lowerlegr", 49),
    ("lowerneck", 4),
    ("lowertorso", 0),
    ("lpinky1", 16),
    ("lpinky2", 17),
    ("lpinky3", 18),
    ("lradius", 8),
    ("lring1", 19),
    ("lring2", 20),
    ("lring3", 21),
    ("lshin", 45),
    ("lshldr", 7),
    ("lshldrbend", 7),
    ("lshoulder", 6),
    ("ltalus", 46),
    ("ltarsus", 46),
    ("lthigh", 44),
    ("lthighbend", 44),
    ("lthumb1", 22),
    ("lthumb2", 23),
    ("lthumb3", 24),
    ("ltibia", 45),
    ("ltoe", 47),
    ("ltoebase", 47),
    ("ltoes", 47),
    ("lulna", 8),
    ("lumbar", 1),
    ("lupleg", 44),
    ("lupperarm", 7),
    ("lupperleg", 44),
    ("lwrist", 9),
    ("mankleleft", 46),
    ("mankleright", 50),
    ("mchest", 2),
    ("mcollarleft", 6),
    ("mcollarright", 25),
    ("melbowleft", 8),
    ("melbowright", 27),
    ("metatarsusl", 47),
    ("metatarsusr", 51),
    ("mfootleft", 47),
    ("mfootright", 51),
    ("mhead", 5),
    ("mhipleft", 44),
    ("mhipright", 48),
    ("middle01l", 13),
    ("middle01r", 32),
    ("middle02l", 14),
    ("middle02r", 33),
    ("middle03l", 15),
    ("middle03r", 34),
    ("middle1l", 13),
    ("middle1r", 32),
    ("middle2l", 14),
    ("middle2r", 33),
    ("middle3l", 15),
    ("middle3r", 34),
    ("middledistall", 15),
    ("middledistalr", 34),
    ("middlefinger1l", 13),
    ("middlefinger1r", 32),
    ("middlefinger2l", 14),
    ("middlefinger2r", 33),
    ("middlefinger3l", 15),
    ("middlefinger3r", 34),
    ("middlefingerl", 13),
    ("middlefingerr", 32),
    ("middleintermediatel", 14),
    ("middleintermediater", 33),
    ("middlel", 13),
    ("middleproximall", 13),
    ("middleproximalr", 32),
    ("middler", 32),
    ("mkneeleft", 45),
    ("mkneeright", 49),
    ("mneck", 4),
    ("mpelvis", 0),
    ("mshoulderleft", 7),
    ("mshoulderright", 26),
    ("mtoeleft", 47),
    ("mtoeright", 51),
    ("mtorso", 1),
    ("mwristleft", 9),
    ("mwristright", 28),
    ("neck", 4),
    ("neck01", 4),
    ("necklower", 4),
    ("neckm", 4),
    ("necktwist01", 4),
    ("necktwist02", 4),
    ("neckupper", 4),
    ("pelvis", 0),
    ("pelvism", 0),
    ("pinky01l", 16),
    ("pinky01r", 35),
    ("pinky02l", 17),
    ("pinky02r", 36),
    ("pinky03l", 18),
    ("pinky03r", 37),
    ("pinky1l", 16),
    ("pinky1r", 35),
    ("pinky2l", 17),
    ("pinky2r", 36),
    ("pinky3l", 18),
    ("pinky3r", 37),
    ("pinkydistall", 18),
    ("pinkydistalr", 37),
    ("pinkyfinger1l", 16),
    ("pinkyfinger1r", 35),
    ("pinkyfinger2l", 17),
    ("pinkyfinger2r", 36),
    ("pinkyfinger3l", 18),
    ("pinkyfinger3r", 37),
    ("pinkyfingerl", 16),
    ("pinkyfingerr", 35),
    ("pinkyintermediatel", 17),
    ("pinkyintermediater", 36),
    ("pinkyl", 16),
    ("pinkyproximall", 16),
    ("pinkyproximalr", 35),
    ("pinkyr", 35),
    ("radiusl", 8),
    ("radiusr", 27),
    ("rankle", 50),
    ("rarm", 26),
    ("rcalcaneus", 50),
    ("rcalf", 49),
    ("rcarpus", 28),
    ("rclavicle", 25),
    ("rcollar", 25),
    ("relbow", 27),
    ("rfemur", 48),
    ("rfibula", 49),
    ("rfoot", 50),
    ("rforearm", 27),
    ("rforearmbend", 27),
    ("rhand", 28),
    ("rhip", 48),
    ("rhumerus", 26),
    ("ribcage", 2),
    ("ribs", 2),
    ("rightankle", 50),
    ("rightarm", 26),
    ("rightcalf", 49),
    ("rightclavicle", 25),
    ("rightcollar", 25),
    ("rightelbow", 27),
    ("rightfoot", 50),
    ("rightforearm", 27),
    ("righthand", 28),
    ("righthandindex1", 29),
    ("righthandindex2", 30),
    ("righthandindex3", 31),
    ("righthandmiddle1", 32),
    ("righthandmiddle2", 33),
    ("righthandmiddle3", 34),
    ("righthandpinky1", 35),
    ("righthandpinky2", 36),
    ("righthandpinky3", 37),
    ("righthandring1", 38),
    ("righthandring2", 39),
    ("righthandring3", 40),
    ("righthandthumb1", 41),
    ("righthandthumb2", 42),
    ("righthandthumb3", 43),
    ("righthip", 48),
    ("rightindexdistal", 31),
    ("rightindexintermediate", 30),
    ("rightindexproximal", 29),
    ("rightknee", 49),
    ("rightleg", 49),
    ("rightlittledistal", 37),
    ("rightlittleintermediate", 36),
    ("rightlittleproximal", 35),
    ("rightlowerarm", 27),
    ("rightlowerleg", 49),
    ("rightmiddledistal", 34),
    ("rightmiddleintermediate", 33),
    ("rightmiddleproximal", 32),
    ("rightpinkydistal", 37),
    ("rightpinkyintermediate", 36),
    ("rightpinkyproximal", 35),
    ("rightringdistal", 40),
    ("rightringintermediate", 39),
    ("rightringproximal", 38),
    ("rightshin", 49),
    ("rightshoulder", 25),
    ("rightthigh", 48),
    ("rightthumbdistal", 43),
    ("rightthumbmetacarpal", 41),
    ("rightthumbproximal", 42),
    ("righttoe", 51),
    ("righttoebase", 51),
    ("righttoes", 51),
    ("rightupleg", 48),
    ("rightupperarm", 26),
    ("rightupperleg", 48),
    ("rightwrist", 28),
    ("rindex1", 29),
    ("rindex2", 30),
    ("rindex3", 31),
    ("ring01l", 19),
    ("ring01r", 38),
    ("ring02l", 20),
    ("ring02r", 39),
    ("ring03l", 21),
    ("ring03r", 40),
    ("ring1l", 19),
    ("ring1r", 38),
    ("ring2l", 20),
    ("ring2r", 39),
    ("ring3l", 21),
    ("ring3r", 40),
    ("ringdistall", 21),
    ("ringdistalr", 40),
    ("ringfinger1l", 19),
    ("ringfinger1r", 38),
    ("ringfinger2l", 20),
    ("ringfinger2r", 39),
    ("ringfinger3l", 21),
    ("ringfinger3r", 40),
    ("ringfingerl", 19),
    ("ringfingerr", 38),
    ("ringintermediatel", 20),
    ("ringintermediater", 39),
    ("ringl", 19),
    ("ringproximall", 19),
    ("ringproximalr", 38),
    ("ringr", 38),
    ("rknee", 49),
    ("rleg", 49),
    ("rlittle1", 35),
    ("rlittle2", 36),
    ("rlittle3", 37),
    ("rlowerarm", 27),
    ("rlowerleg", 49),
    ("rmetatarsus", 51),
    ("rmid1", 32),
    ("rmid2", 33),
    ("rmid3", 34),
    ("rmiddle1", 32),
    ("rmiddle2", 33),
    ("rmiddle3", 34),
    ("rootm", 0),
    ("rpinky1", 35),
    ("rpinky2", 36),
    ("rpinky3", 37),
    ("rradius", 27),
    ("rring1", 38),
    ("rring2", 39),
    ("rring3", 40),
    ("rshin", 49),
    ("rshldr", 26),
    ("rshldrbend", 26),
    ("rshoulder", 25),
    ("rtalus", 50),
    ("rtarsus", 50),
    ("rthigh", 48),
    ("rthighbend", 48),
    ("rthumb1", 41),
    ("rthumb2", 42),
    ("rthumb3", 43),
    ("rtibia", 49),
    ("rtoe", 51),
    ("rtoebase", 51),
    ("rtoes", 51),
    ("rulna", 27),
    ("rupleg", 48),
    ("rupperarm", 26),
    ("rupperleg", 48),
    ("rwrist", 28),
    ("scapulal", 6),
    ("scapular", 25),
    ("shinl", 45),
    ("shinr", 49),
    ("shoulderl", 7),
    ("shoulderr", 26),
    ("skull", 5),
    ("spine", 1),
    ("spine0", 1),
    ("spine1", 2),
    ("spine1m", 1),
    ("spine2", 3),
    ("spine2m", 2),
    ("spine3m", 3),
    ("sternum", 3),
    ("talusl", 46),
    ("talusr", 50),
    ("tarsusl", 46),
    ("tarsusr", 50),
    ("thighl", 44),
    ("thighr", 48),
    ("thoracic", 2),
    ("thumb01l", 22),
    ("thumb01r", 41),
    ("thumb02l", 23),
    ("thumb02r", 42),
    ("thumb03l", 24),
    ("thumb03r", 43),
    ("thumb1l", 22),
    ("thumb1r", 41),
    ("thumb2l", 23),
    ("thumb2r", 42),
    ("thumb3l", 24),
    ("thumb3r", 43),
    ("thumbdistall", 24),
    ("thumbdistalr", 43),
    ("thumbfinger1l", 22),
    ("thumbfinger1r", 41),
    ("thumbfinger2l", 23),
    ("thumbfinger2r", 42),
    ("thumbfinger3l", 24),
    ("thumbfinger3r", 43),
    ("thumbfingerl", 22),
    ("thumbfingerr", 41),
    ("thumbl", 22),
    ("thumbmetacarpall", 22),
    ("thumbmetacarpalr", 41),
    ("thumbproximall", 23),
    ("thumbproximalr", 42),
    ("thumbr", 41),
    ("tibial", 45),
    ("tibiar", 49),
    ("toel", 47),
    ("toer", 51),
    ("torso", 1),
    ("torsolower", 0),
    ("torsoupper", 2),
    ("ulnal", 8),
    ("ulnar", 27),
    ("upperarml", 7),
    ("upperarmr", 26),
    ("upperchest", 3),
    ("upperlegl", 44),
    ("upperlegr", 48),
    ("upperneck", 4),
    ("uppertorso", 2),
    ("waist", 1),
    ("wristl", 9),
    ("wristr", 28),
    ("センター", 0),
    ("上半身", 1),
    ("上半身2", 2),
    ("下半身", 0),
    ("右つま先", 51),
    ("右ひざ", 49),
    ("右ひじ", 27),
    ("右中指1", 32),
    ("右中指2", 33),
    ("右中指3", 34),
    ("右人差指1", 29),
    ("右人差指2", 30),
    ("右人差指3", 31),
    ("右人指1", 29),
    ("右人指2", 30),
    ("右人指3", 31),
    ("右小指1", 35),
    ("右小指2", 36),
    ("右小指3", 37),
    ("右手首", 28),
    ("右肩", 25),
    ("右腕", 26),
    ("右薬指1", 38),
    ("右薬指2", 39),
    ("右薬指3", 40),
    ("右親指0", 41),
    ("右親指1", 42),
    ("右親指2", 43),
    ("右足", 48),
    ("右足首", 50),
    ("左つま先", 47),
    ("左ひざ", 45),
    ("左ひじ", 8),
    ("左中指1", 13),
    ("左中指2", 14),
    ("左中指3", 15),
    ("左人差指1", 10),
    ("左人差指2", 11),
    ("左人差指3", 12),
    ("左人指1", 10),
    ("左人指2", 11),
    ("左人指3", 12),
    ("左小指1", 16),
    ("左小指2", 17),
    ("左小指3", 18),
    ("左手首", 9),
    ("左肩", 6),
    ("左腕", 7),
    ("左薬指1", 19),
    ("左薬指2", 20),
    ("左薬指3", 21),
    ("左親指0", 22),
    ("左親指1", 23),
    ("左親指2", 24),
    ("左足", 44),
    ("左足首", 46),
    ("頭", 5),
    ("首", 4),
];

/// Canonical name of a bone index.
pub fn name(index: usize) -> &'static str {
    BONES[index.min(BONE_COUNT - 1)]
}

/// Resolve one rig bone name to a canonical bone.
///
/// Mirrors `canonicalizeBoneName`: a direct lookup, then one retry with a
/// trailing `.1` / `_1` duplicate suffix removed.
pub fn canonicalize(raw: &str) -> Option<usize> {
    if raw.is_empty() {
        return None;
    }
    if let Some(found) = lookup(raw) {
        return Some(found);
    }
    let trimmed = strip_duplicate_index(raw);
    if trimmed.len() != raw.len() {
        lookup(trimmed)
    } else {
        None
    }
}

/// Share of `bones` that resolves to the canonical set (`0..=1`).
pub fn coverage<'a>(bones: impl IntoIterator<Item = &'a str>) -> f32 {
    let mut total = 0usize;
    let mut matched = 0usize;
    for bone in bones {
        total += 1;
        if canonicalize(bone).is_some() {
            matched += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        matched as f32 / total as f32
    }
}

fn lookup(raw: &str) -> Option<usize> {
    let mut name = strip_namespace(raw);
    name = strip_mixamo(name);
    name = strip_prefix_optional(name, "Armature", &['_', '/']);
    name = strip_tag(name);
    name = strip_prefix_required(name, "CH", &['_', ':']);
    name = strip_prefix_plain(name, "CC_Base_");
    name = strip_low_j(name);
    name = strip_numeric_dot(name);
    name = strip_prefix_plain(name, "!");
    name = strip_prefix_required(name, "joint", &['_', ':']);
    name = strip_biped(name);
    let key: String = name
        .chars()
        .filter(|c| !c.is_whitespace() && !matches!(c, '-' | '_' | '.'))
        .flat_map(|c| c.to_lowercase())
        // MMD writes its finger segments with full-width digits (`左人指１`).
        .map(|c| match c {
            '\u{ff10}'..='\u{ff19}' => char::from(b'0' + (c as u32 - 0xff10) as u8),
            '\u{ff21}'..='\u{ff3a}' => char::from(b'a' + (c as u32 - 0xff21) as u8),
            other => other,
        })
        .collect();
    ALIASES
        .iter()
        .chain(EXTRA.iter())
        .find(|(alias, _)| *alias == key)
        .map(|(_, index)| *index as usize)
}

/// `^(?:[A-Za-z][\w]*:)+` — drop every leading `namespace:` group.
fn strip_namespace(raw: &str) -> &str {
    let mut rest = raw;
    loop {
        let bytes = rest.as_bytes();
        if !bytes.first().map(|b| b.is_ascii_alphabetic()).unwrap_or(false) {
            return rest;
        }
        let mut index = 1;
        while index < bytes.len() && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_') {
            index += 1;
        }
        if index < bytes.len() && bytes[index] == b':' {
            rest = &rest[index + 1..];
            continue;
        }
        return rest;
    }
}

/// `^mixamorig\d*[_:]?`
fn strip_mixamo<'a>(raw: &'a str) -> &'a str {
    let prefix = "mixamorig";
    if raw.len() < prefix.len() || !raw.is_char_boundary(prefix.len()) {
        return raw;
    }
    if !raw[..prefix.len()].eq_ignore_ascii_case(prefix) {
        return raw;
    }
    let mut rest = &raw[prefix.len()..];
    rest = rest.trim_start_matches(|c: char| c.is_ascii_digit());
    if let Some(first) = rest.chars().next() {
        if first == '_' || first == ':' {
            rest = &rest[first.len_utf8()..];
        }
    }
    rest
}

/// `PREFIX` followed by any one of `extra`; the separator is optional.
fn strip_prefix_optional<'a>(raw: &'a str, prefix: &str, extra: &[char]) -> &'a str {
    match strip_ci_start(raw, prefix) {
        Some(rest) => match rest.chars().next() {
            Some(c) if extra.contains(&c) => &rest[c.len_utf8()..],
            _ => rest,
        },
        None => raw,
    }
}

/// `PREFIX` followed by one of `extra`; the separator is required.
fn strip_prefix_required<'a>(raw: &'a str, prefix: &str, extra: &[char]) -> &'a str {
    match strip_ci_start(raw, prefix) {
        Some(rest) => match rest.chars().next() {
            Some(c) if extra.contains(&c) => &rest[c.len_utf8()..],
            _ => raw,
        },
        None => raw,
    }
}

/// `PREFIX` with no separator rule.
fn strip_prefix_plain<'a>(raw: &'a str, prefix: &str) -> &'a str {
    strip_ci_start(raw, prefix).unwrap_or(raw)
}

/// Case-insensitive `prefix`, without allocating.
///
/// The length check is in bytes and the slice is guarded by `is_char_boundary`:
/// an ASCII prefix can only match at a character boundary, and MMD's names are
/// Japanese, where a plain `split_at(prefix.len())` would panic mid-character.
fn strip_ci_start<'a>(raw: &'a str, prefix: &str) -> Option<&'a str> {
    if raw.len() < prefix.len() || !raw.is_char_boundary(prefix.len()) {
        return None;
    }
    let (head, rest) = raw.split_at(prefix.len());
    head.eq_ignore_ascii_case(prefix).then_some(rest)
}

/// `^(DEF|ORG|MCH)[-_]`
fn strip_tag<'a>(raw: &'a str) -> &'a str {
    for tag in ["DEF-", "DEF_", "ORG-", "ORG_", "MCH-", "MCH_"] {
        if let Some(rest) = strip_ci_start(raw, tag) {
            return rest;
        }
    }
    raw
}

/// `^j_(?!bip|sec|adj)` — VRM's `J_Bip_*` survives, other `j_` does not.
fn strip_low_j<'a>(raw: &'a str) -> &'a str {
    let Some(rest) = strip_ci_start(raw, "j_") else {
        return raw;
    };
    for keep in ["bip", "sec", "adj"] {
        if rest.len() >= keep.len() && rest[..keep.len()].eq_ignore_ascii_case(keep) {
            return raw;
        }
    }
    rest
}

/// `^\d+\.`
fn strip_numeric_dot<'a>(raw: &'a str) -> &'a str {
    let digits = raw.len() - raw.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits > 0 && raw[digits..].starts_with('.') {
        &raw[digits + 1..]
    } else {
        raw
    }
}

/// `^Bip\d+[\s_]?`
fn strip_biped<'a>(raw: &'a str) -> &'a str {
    let Some(rest) = strip_ci_start(raw, "Bip") else {
        return raw;
    };
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 {
        return raw;
    }
    let mut tail = &rest[digits..];
    if let Some(first) = tail.chars().next() {
        if first == '_' || first.is_whitespace() {
            tail = &tail[first.len_utf8()..];
        }
    }
    tail
}

/// `[._]\d+$`
fn strip_duplicate_index(raw: &str) -> &str {
    let digits = raw.len() - raw.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 || digits >= raw.len() {
        return raw;
    }
    let before = &raw[..raw.len() - digits];
    match before.chars().last() {
        Some('.') | Some('_') => &raw[..before.len() - 1],
        _ => raw,
    }
}
// ---------------------------------------------------------------------------
// Export side: our own names, the vendor tables, and the rest correction.
// ---------------------------------------------------------------------------

/// Our exporter's own bone names, in canonical order.
///
/// These are the Character Studio biped names `flow::prepare_bvh` writes
/// (`Chest`, `LeftHip`, `LeftFinger10` …): the reference application exports a
/// Mixamo biped and renames it to this vocabulary. Indexing by canonical bone
/// keeps the table readable and lets [`from_export_name`] be a plain position
/// lookup.
///
/// Note one deliberate difference from the package's alias table: it reads a
/// bare `chest` as `Spine1`, ours is the first spine bone above the pelvis
/// (`Spine`), because the name comes from the SMPL-H joint it was derived from
/// rather than from the alias. `Chest2`/`Chest3` have no alias at all.
pub const BIPED: [&str; BONE_COUNT] = [
    "Hips",          // Hips
    "Chest",         // Spine
    "Chest2",        // Spine1
    "Chest3",        // Spine2
    "Neck",          // Neck
    "Head",          // Head
    "LeftCollar",    // LeftShoulder
    "LeftShoulder",  // LeftArm
    "LeftElbow",     // LeftForeArm
    "LeftWrist",     // LeftHand
    "LeftFinger1",   // LeftHandIndex1
    "LeftFinger11",  // LeftHandIndex2
    "LeftFinger12",  // LeftHandIndex3
    "LeftFinger2",   // LeftHandMiddle1
    "LeftFinger21",  // LeftHandMiddle2
    "LeftFinger22",  // LeftHandMiddle3
    "LeftFinger4",   // LeftHandPinky1
    "LeftFinger41",  // LeftHandPinky2
    "LeftFinger42",  // LeftHandPinky3
    "LeftFinger3",   // LeftHandRing1
    "LeftFinger31",  // LeftHandRing2
    "LeftFinger32",  // LeftHandRing3
    "LeftFinger0",   // LeftHandThumb1
    "LeftFinger01",  // LeftHandThumb2
    "LeftFinger02",  // LeftHandThumb3
    "RightCollar",   // RightShoulder
    "RightShoulder", // RightArm
    "RightElbow",    // RightForeArm
    "RightWrist",    // RightHand
    "RightFinger1",  // RightHandIndex1
    "RightFinger11", // RightHandIndex2
    "RightFinger12", // RightHandIndex3
    "RightFinger2",  // RightHandMiddle1
    "RightFinger21", // RightHandMiddle2
    "RightFinger22", // RightHandMiddle3
    "RightFinger4",  // RightHandPinky1
    "RightFinger41", // RightHandPinky2
    "RightFinger42", // RightHandPinky3
    "RightFinger3",  // RightHandRing1
    "RightFinger31", // RightHandRing2
    "RightFinger32", // RightHandRing3
    "RightFinger0",  // RightHandThumb1
    "RightFinger01", // RightHandThumb2
    "RightFinger02", // RightHandThumb3
    "LeftHip",       // LeftUpLeg
    "LeftKnee",      // LeftLeg
    "LeftAnkle",     // LeftFoot
    "LeftToe",       // LeftToeBase
    "RightHip",      // RightUpLeg
    "RightKnee",     // RightLeg
    "RightAnkle",    // RightFoot
    "RightToe",      // RightToeBase
];

/// The naming an export is written in.
///
/// [`Rig::Biped`] is the exporter's default. Every other variant is a real
/// convention whose names are taken from the package's alias table, so each one
/// resolves back to the same canonical bone through [`canonicalize`]: that
/// round trip is what makes a file usable by the package's own retargeter, and
/// the test module asserts it for all 52 bones of every rig.
///
/// Vendors the package handles but this list leaves out, and why:
///
/// * **MakeHuman** — its default skeleton numbers every bone (`spine01`,
///   `upperarm01.L`), which the alias table resolves only in part.
/// * **Rigify** — `spine.001` and friends have no alias either; Blender
///   imports a Mixamo-named BVH and retargets from that.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Rig {
    /// `Hips`, `Chest`, `LeftHip`, `LeftFinger10` — 3ds Max / Character Studio.
    #[default]
    Biped,
    /// The canonical Mixamo names (`Hips`, `Spine2`, `LeftUpLeg`).
    Mixamo,
    /// The same, behind the namespace Mixamo's own FBX files carry.
    MixamoPrefixed,
    /// VRM 0.x / 1.0 humanoid joints (`J_Bip_L_UpperArm`).
    Vrm,
    /// Unreal's mannequin (`pelvis`, `spine_01`, `upperarm_l`, `ball_l`).
    Unreal,
    /// DAZ Genesis (`hip`, `lShldr`, `lThigh`, `lIndex1`).
    Daz,
    /// MMD / PMX standard bones (`センター`, `上半身`, `左腕`, `左ひじ`,
    /// `左親指0`). The standard rig has two spine bones where the canonical
    /// skeleton has three, so the third is written as `上半身3` — the
    /// continuation MMD models use when they carry one — and `EXTRA` resolves
    /// it back. Its bind pose is per model, so the rest correction is off by
    /// default like Unreal's.
    Mmd,
}

impl Rig {
    pub const ALL: [Rig; 7] = [
        Rig::Biped,
        Rig::Mixamo,
        Rig::MixamoPrefixed,
        Rig::Vrm,
        Rig::Unreal,
        Rig::Daz,
        Rig::Mmd,
    ];

    /// Interface label.
    pub fn label(self) -> &'static str {
        match self {
            Rig::Biped => "VeloMocap biped",
            Rig::Mixamo => "Mixamo",
            Rig::MixamoPrefixed => "Mixamo (mixamorig:)",
            Rig::Vrm => "VRM",
            Rig::Unreal => "Unreal",
            Rig::Daz => "DAZ Genesis",
            Rig::Mmd => "MMD / PMX",
        }
    }

    /// Command-line spelling.
    pub fn slug(self) -> &'static str {
        match self {
            Rig::Biped => "biped",
            Rig::Mixamo => "mixamo",
            Rig::MixamoPrefixed => "mixamo-prefixed",
            Rig::Vrm => "vrm",
            Rig::Unreal => "unreal",
            Rig::Daz => "daz",
            Rig::Mmd => "mmd",
        }
    }

    pub fn from_slug(raw: &str) -> Option<Rig> {
        Rig::ALL
            .into_iter()
            .find(|rig| rig.slug().eq_ignore_ascii_case(raw.trim()))
    }

    /// Whether this rig binds in the canonical T-pose, i.e. whether the
    /// rest-pose correction is the right default for it.
    ///
    /// The canonical rest is Mixamo's, so Mixamo, VRM and DAZ (all T-pose
    /// binds) line up with it. The Unreal mannequin binds in an A-pose, so the
    /// correction would rotate every arm by 45°, and an MMD model's bind is
    /// whatever the PMX author made it — neither is a T-pose by contract, so
    /// both default off.
    pub fn binds_in_t_pose(self) -> bool {
        !matches!(self, Rig::Unreal | Rig::Mmd)
    }

    /// Whether this rig spells its bones in ASCII, i.e. whether the FBX writer
    /// accepts them. Only MMD does not: its standard bones are Japanese, and the
    /// FBX importers this has been checked against are not reliable about
    /// non-ASCII bone names.
    pub fn ascii_names(self) -> bool {
        !matches!(self, Rig::Mmd)
    }

    /// The name this rig gives a canonical bone.
    pub fn name(self, bone: usize) -> String {
        let bone = bone.min(BONE_COUNT - 1);
        match self {
            Rig::Biped => BIPED[bone].to_string(),
            Rig::Mixamo => BONES[bone].to_string(),
            Rig::MixamoPrefixed => format!("mixamorig:{}", BONES[bone]),
            Rig::Vrm => vrm_name(bone),
            Rig::Unreal => unreal_name(bone),
            Rig::Daz => daz_name(bone),
            Rig::Mmd => mmd_name(bone),
        }
    }
}

/// Canonical bone of one of our own exported names.
fn biped_index(raw: &str) -> Option<usize> {
    BIPED.iter().position(|name| *name == raw)
}

/// Canonical bone of one of our exported names, or of any vendor name the
/// package resolves.
///
/// The two vocabularies overlap on `LeftShoulder` and `RightShoulder`: ours is
/// the upper arm (and `LeftCollar`/`RightCollar` are the clavicle), Mixamo's is
/// the clavicle. The biped reading wins here, because this is the name this
/// crate *writes*; a clip that came from a vendor rig has to be resolved with
/// [`canonicalize`] instead — which is what [`rewrite`] does once it sees that
/// the names are not ours.
pub fn from_export_name(raw: &str) -> Option<usize> {
    biped_index(raw).or_else(|| canonicalize(raw))
}

/// Side, digit (`0` thumb … `4` pinky, Character Studio order) and segment
/// (`1..=3`) of a finger bone.
fn finger_parts(bone: usize) -> Option<(char, usize, usize)> {
    // Canonical order per hand: index, middle, pinky, ring, thumb — CS digit
    // order is thumb, index, middle, ring, pinky.
    let (base, digits) = if (10..25).contains(&bone) {
        ('L', [10u8, 13, 16, 19, 22])
    } else if (29..44).contains(&bone) {
        ('R', [29u8, 32, 35, 38, 41])
    } else {
        return None;
    };
    for (digit, start) in [1usize, 2, 4, 3, 0].into_iter().zip(digits) {
        let start = start as usize;
        if bone < start + 3 {
            return Some((base, digit, bone - start + 1));
        }
    }
    None
}

fn vrm_name(bone: usize) -> String {
    if let Some((side, digit, segment)) = finger_parts(bone) {
        let finger = ["Thumb", "Index", "Middle", "Ring", "Little"][digit];
        return format!("J_Bip_{side}_{finger}{segment}");
    }
    let (side, part) = match bone {
        0 => ('C', "Hips"),
        1 => ('C', "Spine"),
        2 => ('C', "Chest"),
        3 => ('C', "UpperChest"),
        4 => ('C', "Neck"),
        5 => ('C', "Head"),
        6 => ('L', "Shoulder"),
        7 => ('L', "UpperArm"),
        8 => ('L', "LowerArm"),
        9 => ('L', "Hand"),
        25 => ('R', "Shoulder"),
        26 => ('R', "UpperArm"),
        27 => ('R', "LowerArm"),
        28 => ('R', "Hand"),
        44 => ('L', "UpperLeg"),
        45 => ('L', "LowerLeg"),
        46 => ('L', "Foot"),
        47 => ('L', "ToeBase"),
        48 => ('R', "UpperLeg"),
        49 => ('R', "LowerLeg"),
        50 => ('R', "Foot"),
        51 => ('R', "ToeBase"),
        _ => return BONES[bone].to_string(),
    };
    format!("J_Bip_{side}_{part}")
}

fn unreal_name(bone: usize) -> String {
    if let Some((side, digit, segment)) = finger_parts(bone) {
        let finger = ["thumb", "index", "middle", "ring", "pinky"][digit];
        let side = if side == 'L' { "l" } else { "r" };
        return format!("{finger}_0{segment}_{side}");
    }
    // UE5's spine chain has no package alias, and the package's dedupe fallback
    // would collapse all three onto `Spine`; `EXTRA` names them in order.
    match bone {
        0 => "pelvis".to_string(),
        1 => "spine_01".to_string(),
        2 => "spine_02".to_string(),
        3 => "spine_03".to_string(),
        4 => "neck_01".to_string(),
        5 => "head".to_string(),
        6 => "clavicle_l".to_string(),
        7 => "upperarm_l".to_string(),
        8 => "lowerarm_l".to_string(),
        9 => "hand_l".to_string(),
        25 => "clavicle_r".to_string(),
        26 => "upperarm_r".to_string(),
        27 => "lowerarm_r".to_string(),
        28 => "hand_r".to_string(),
        44 => "thigh_l".to_string(),
        45 => "calf_l".to_string(),
        46 => "foot_l".to_string(),
        47 => "ball_l".to_string(),
        48 => "thigh_r".to_string(),
        49 => "calf_r".to_string(),
        50 => "foot_r".to_string(),
        51 => "ball_r".to_string(),
        other => BONES[other].to_string(),
    }
}

fn daz_name(bone: usize) -> String {
    if let Some((side, digit, segment)) = finger_parts(bone) {
        let finger = ["Thumb", "Index", "Mid", "Ring", "Pinky"][digit];
        let side = if side == 'L' { "l" } else { "r" };
        return format!("{side}{finger}{segment}");
    }
    match bone {
        0 => "hip".to_string(),
        1 => "abdomenLower".to_string(),
        2 => "abdomenUpper".to_string(),
        3 => "upperChest".to_string(),
        4 => "neckLower".to_string(),
        5 => "head".to_string(),
        6 => "lCollar".to_string(),
        7 => "lShldr".to_string(),
        8 => "lForeArm".to_string(),
        9 => "lHand".to_string(),
        25 => "rCollar".to_string(),
        26 => "rShldr".to_string(),
        27 => "rForeArm".to_string(),
        28 => "rHand".to_string(),
        44 => "lThigh".to_string(),
        45 => "lShin".to_string(),
        46 => "lFoot".to_string(),
        47 => "lToe".to_string(),
        48 => "rThigh".to_string(),
        49 => "rShin".to_string(),
        50 => "rFoot".to_string(),
        51 => "rToe".to_string(),
        other => BONES[other].to_string(),
    }
}

/// MMD's standard bones, in Japanese, as PMX files spell them.
///
/// The thumb is numbered from 0 in MMD (its metacarpal is `親指0`), the other
/// four fingers from 1; a model without `上半身3` still imports, the bone is just
/// left unmapped, which is how MMD itself treats a three-segment spine.
fn mmd_name(bone: usize) -> String {
    if let Some((side, digit, segment)) = finger_parts(bone) {
        let finger = ["親指", "人指", "中指", "薬指", "小指"][digit];
        let side = if side == 'L' { "左" } else { "右" };
        let index = if digit == 0 { segment - 1 } else { segment };
        return format!("{side}{finger}{index}");
    }
    let (side, part) = match bone {
        0 => ('C', "センター"),
        1 => ('C', "上半身"),
        2 => ('C', "上半身2"),
        3 => ('C', "上半身3"),
        4 => ('C', "首"),
        5 => ('C', "頭"),
        6 => ('L', "肩"),
        7 => ('L', "腕"),
        8 => ('L', "ひじ"),
        9 => ('L', "手首"),
        25 => ('R', "肩"),
        26 => ('R', "腕"),
        27 => ('R', "ひじ"),
        28 => ('R', "手首"),
        44 => ('L', "足"),
        45 => ('L', "ひざ"),
        46 => ('L', "足首"),
        47 => ('L', "つま先"),
        48 => ('R', "足"),
        49 => ('R', "ひざ"),
        50 => ('R', "足首"),
        51 => ('R', "つま先"),
        other => return BONES[other].to_string(),
    };
    match side {
        'C' => part.to_string(),
        'L' => format!("左{part}"),
        _ => format!("右{part}"),
    }
}

/// Rotation matrix of a unit quaternion `[x, y, z, w]`, row-major.
fn quat_matrix(quaternion: &[f32; 4]) -> [f64; 9] {
    let [x, y, z, w] = *quaternion;
    let (x, y, z, w) = (x as f64, y as f64, z as f64, w as f64);
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

fn matrix_multiply(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0f64; 9];
    for row in 0..3 {
        for column in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += a[row * 3 + k] * b[k * 3 + column];
            }
            out[row * 3 + column] = acc;
        }
    }
    out
}

const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// Rest rotation of every bone in canonical world space, row-major.
///
/// Composed the way three.js composes a chain, `world[child] = world[parent] ·
/// local[child]`, from [`REST`] and [`PARENTS`] — which reproduces the
/// package's own `CANONICAL_REST_WORLD` table exactly, so it is not copied out
/// here a second time. Every parent in the canonical order precedes its child,
/// so one forward pass is enough.
pub fn world_rest() -> &'static [[f64; 9]; BONE_COUNT] {
    static CACHE: std::sync::OnceLock<[[f64; 9]; BONE_COUNT]> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        let mut world = [IDENTITY; BONE_COUNT];
        for bone in 0..BONE_COUNT {
            let local = quat_matrix(&REST[bone]);
            world[bone] = match PARENTS[bone] {
                None => local,
                Some(parent) => matrix_multiply(&world[parent as usize], &local),
            };
        }
        world
    })
}

/// Re-express one local rotation in the canonical rest frames.
///
/// The package corrects a track from skeleton `s` to skeleton `t` with
/// `q' = L · q · R`, `L = rest_t · rest_s⁻¹` (world rests when it has them,
/// which is where the `q' = L·q·R` of the header comes from). Our exporter
/// writes identity rests, so with the canonical skeleton as the target
/// `L = parent(world)⁻¹` and `R = bone(world)`, i.e. the target rig ends up
/// rotating the bone to exactly the world orientation this clip specifies.
///
/// At rest (`local` identity) the result is the canonical *local* rest, which
/// is what makes a rest frame stay a rest frame — pinned by the tests.
fn correct_local(local: &[f64; 9], parent_world: &[f64; 9], world: &[f64; 9]) -> [f64; 9] {
    // Association is free, so the two multiplications happen as parent_worldᵀ ·
    // (local · world); the transpose is a plain reindex because a rest rotation
    // is orthonormal.
    let inner = matrix_multiply(local, world);
    let mut out = [0.0f64; 9];
    for row in 0..3 {
        for column in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += parent_world[k * 3 + row] * inner[k * 3 + column];
            }
            out[row * 3 + column] = acc;
        }
    }
    out
}

/// Share of the canonical skeleton a clip covers (`0..=1`).
pub fn bone_coverage(bones: usize) -> f32 {
    (bones.min(BONE_COUNT) as f32) / (BONE_COUNT as f32)
}

/// Rename a prepared clip into `rig`, optionally re-expressing its rotations in
/// the canonical rest frames.
///
/// The hierarchy is untouched: our skeleton and the canonical one are the same
/// 52-bone tree, so a name is all a vendored rig needs. Returns the share of the
/// canonical skeleton the clip fills; a clip that covers no bone at all (the
/// MHR path exports `mhrNNN` names) is refused instead of half-renamed.
///
/// The clip's names are read as the vocabulary they are: all-biped means it came
/// from our exporter ([`BIPED`]), anything else goes through [`canonicalize`].
/// The distinction matters for `LeftShoulder`, which means the upper arm in one
/// and the clavicle in the other.
pub fn rewrite(
    prepared: &mut crate::flow::PreparedBvh,
    rig: Rig,
    rest_correct: bool,
) -> anyhow::Result<f32> {
    use anyhow::Context;

    let bones = prepared.parents.len();
    anyhow::ensure!(bones > 0, "the clip has no bones");
    anyhow::ensure!(
        prepared.names.len() == bones,
        "the clip names {} of its {bones} bones",
        prepared.names.len()
    );
    anyhow::ensure!(
        prepared.local.len() % (bones * 9) == 0,
        "the clip's rotation track is truncated"
    );

    let own = prepared.names.iter().all(|name| biped_index(name).is_some());
    let mut canonical = Vec::with_capacity(bones);
    for name in &prepared.names {
        let index = if own {
            biped_index(name)
        } else {
            canonicalize(name)
        }
        .with_context(|| format!("`{name}` is not a bone of the canonical skeleton"))?;
        anyhow::ensure!(
            !canonical.contains(&index),
            "`{name}` names {} twice",
            BONES[index]
        );
        canonical.push(index);
    }

    if rest_correct {
        // The correction composes the parent's world rest, so the hierarchy has
        // to be the canonical one: a collar that hangs off the spine in one and
        // off the chest in the other would rotate with the wrong frame. The
        // canonical bone of the parent has to be translated back into a joint of
        // *this* clip before it can be compared.
        let mut joint_of = [usize::MAX; BONE_COUNT];
        for (joint, bone) in canonical.iter().enumerate() {
            joint_of[*bone] = joint;
        }
        for (joint, index) in canonical.iter().enumerate() {
            let expected = prepared.parents[joint];
            let actual = PARENTS[*index]
                .map(|parent| joint_of[parent as usize] as i32)
                .unwrap_or(-1);
            anyhow::ensure!(
                expected == actual,
                "{} hangs off the wrong bone in this skeleton",
                BONES[*index]
            );
        }
    }

    prepared.names = canonical.iter().map(|bone| rig.name(*bone)).collect();

    if rest_correct {
        let world = world_rest();
        let frames = prepared.frames();
        let mut corrected = vec![0.0f64; prepared.local.len()];
        for frame in 0..frames {
            for joint in 0..bones {
                let at = (frame * bones + joint) * 9;
                let local: [f64; 9] = std::array::from_fn(|i| prepared.local[at + i]);
                let parent = prepared.parents[joint];
                let parent_world = if parent < 0 {
                    IDENTITY
                } else {
                    world[canonical[parent as usize]]
                };
                let own = &world[canonical[joint]];
                let corrected_local = correct_local(&local, &parent_world, own);
                corrected[at..at + 9].copy_from_slice(&corrected_local);
            }
        }
        prepared.local = corrected;
    }

    Ok(bone_coverage(canonical.len()))
}

/// Alias corrections for names the package does not resolve.
///
/// * **Unreal's spine.** The package has no entry for `spine_01..03`, so its
///   duplicate-suffix fallback (`/[._]\d+$/`) strips the index and returns
///   `Spine` for all three — Unreal's chest collapses onto its waist.
/// * **MMD's third spine.** The standard rig stops at `上半身2`; `上半身3` is the
///   name a PMX uses when it carries one, and our skeleton has three spine
///   bones to write. A model without it simply leaves that joint unmapped.
///
/// Both are looked up in the direct pass, before the fallback, so no name that
/// the package already resolved changes meaning.
const EXTRA: &[(&str, u8)] = &[
    ("spine01", 1),
    ("spine02", 2),
    ("spine03", 3),
    ("上半身3", 3),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The package's canonical names must round-trip, or a rig that is already
    /// canonical would be mangled on the way in.
    #[test]
    fn every_canonical_name_resolves_to_itself() {
        for (index, bone) in BONES.iter().enumerate() {
            assert_eq!(canonicalize(bone), Some(index), "canonical name {bone}");
        }
    }

    /// One bone per supported rig convention: prefix, VRM, Unreal, generic
    /// `upperarm_r`-style, tagged Blender (`DEF-`), duplicate suffix.
    #[test]
    fn vendor_conventions_resolve_to_the_same_bone() {
        let cases: [(&str, &str); 10] = [
            ("mixamorig:LeftArm", "LeftArm"),
            ("mixamorig1:LeftArm", "LeftArm"),
            ("J_Bip_L_UpperArm", "LeftArm"),
            ("pelvis", "Hips"),
            ("clavicle_l", "LeftShoulder"),
            ("upperarm_r", "RightArm"),
            ("thigh_l", "LeftUpLeg"),
            ("DEF-spine", "Spine"),
            ("LeftUpLeg.001", "LeftUpLeg"),
            ("Bip01 L Foot", "LeftFoot"),
        ];
        for (raw, expected) in cases {
            assert_eq!(
                canonicalize(raw).map(name),
                Some(expected),
                "{raw} did not resolve to {expected}"
            );
        }
    }

    /// Garbage in, `None` out — including names that are not humanoid bones.
    #[test]
    fn unknown_names_are_rejected() {
        for raw in ["", "Camera", "twist_01", "pSphere1", "wand"] {
            assert_eq!(canonicalize(raw), None, "{raw} should not resolve");
        }
    }

    /// A duplicate alias would silently shadow another bone: the table is
    /// generated, so this pins the generation.
    #[test]
    fn aliases_are_unique_and_in_range() {
        let mut seen = std::collections::HashSet::new();
        for (alias, index) in ALIASES {
            assert!(*index < BONE_COUNT as u8, "alias {alias} -> {index}");
            assert!(!alias.is_empty(), "empty alias");
            assert!(seen.insert(*alias), "duplicate alias {alias}");
        }
    }

    /// One bone per vendor shares the canonical order, so the hierarchy is a
    /// single tree rooted at `Hips`.
    #[test]
    fn the_skeleton_is_one_tree() {
        let mut roots = 0;
        for (index, parent) in PARENTS.iter().enumerate() {
            match parent {
                None => roots += 1,
                Some(parent) => {
                    assert!((*parent as usize) < BONE_COUNT, "{} parent out of range", BONES[index]);
                    assert_ne!(*parent as usize, index, "{} is its own parent", BONES[index]);
                }
            }
        }
        assert_eq!(roots, 1, "expected exactly one root");
        assert_eq!(PARENTS[0], None, "Hips is the root");
        // Every bone must reach the root without a cycle.
        for index in 0..BONE_COUNT {
            let mut cursor = Some(index);
            let mut steps = 0;
            while let Some(at) = cursor {
                cursor = PARENTS[at].map(|parent| parent as usize);
                steps += 1;
                assert!(steps <= BONE_COUNT, "cycle at {}", BONES[index]);
            }
        }
    }

    /// The rest quaternions are used for the retarget correction, so they have
    /// to be unit quaternions.
    #[test]
    fn rest_quaternions_are_unit() {
        for (index, rest) in REST.iter().enumerate() {
            let norm: f32 = rest.iter().map(|value| value * value).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "{} rest quaternion norm {norm}",
                BONES[index]
            );
        }
    }

    /// The coverage gate: a full canonical rig passes, an empty one does not.
    #[test]
    fn coverage_matches_the_package_gate() {
        assert_eq!(coverage(BONES.iter().copied()), 1.0);
        assert_eq!(coverage(std::iter::empty()), 0.0);
        let mixamo: Vec<&str> = BONES.iter().map(|bone| bone).copied().collect();
        assert!(coverage(mixamo) >= MIN_COVERAGE);
        let junk = ["cube", "lamp", "camera"].into_iter();
        assert!(coverage(junk) < MIN_COVERAGE);
    }

    // -----------------------------------------------------------------------
    // The export chain.
    // -----------------------------------------------------------------------

    /// SMPL-H's hierarchy, in the joint order the exporter walks (`pelvis`,
    /// `left_hip`, `right_hip`, `spine1`, …). This is the rig every clip of ours
    /// carries; the names come from the exporter itself, so what the tests below
    /// check is the mapping from *that* skeleton onto the canonical one.
    const SMPL_PARENTS: [i32; BONE_COUNT] = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 20,
        25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47,
        21, 49, 50,
    ];

    fn identity_motion(_bone: usize) -> [f64; 9] {
        IDENTITY
    }

    /// One frame of the exporter's own payload: its names, the SMPL-H
    /// hierarchy, and whatever local rotations the test wants per joint.
    fn fixture(local: impl Fn(usize) -> [f64; 9]) -> crate::flow::PreparedBvh {
        let bones = BONE_COUNT;
        let mut rotations = vec![0.0f64; bones * 9];
        for joint in 0..bones {
            let m = local(joint);
            rotations[joint * 9..joint * 9 + 9].copy_from_slice(&m);
        }
        crate::flow::PreparedBvh {
            names: (0..bones).map(crate::flow::biped_name).collect(),
            parents: SMPL_PARENTS.to_vec(),
            rest: vec![0.0; bones * 3],
            roots: vec![0.0, 0.0, 0.0],
            local: rotations,
            fps: 30.0,
        }
    }

    /// The same, over `frames` frames, so the correction's frame stride is
    /// exercised rather than assumed.
    fn fixture_frames(
        frames: usize,
        local: impl Fn(usize, usize) -> [f64; 9],
    ) -> crate::flow::PreparedBvh {
        let bones = BONE_COUNT;
        let mut prepared = fixture(|_| IDENTITY);
        prepared.local = vec![0.0f64; frames * bones * 9];
        for frame in 0..frames {
            for joint in 0..bones {
                let at = (frame * bones + joint) * 9;
                prepared.local[at..at + 9].copy_from_slice(&local(frame, joint));
            }
        }
        prepared.roots = vec![0.0; frames * 3];
        prepared
    }

    /// A different rotation on every bone, so a correction that reuses one
    /// frame everywhere would not pass.
    fn turned(bone: usize) -> [f64; 9] {
        let angle = 0.05 + bone as f64 * 0.11;
        let axis = [0.3 + bone as f64 * 0.01, 0.5, 0.7];
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let (x, y, z) = (axis[0] / norm, axis[1] / norm, axis[2] / norm);
        let (s, c) = (angle.sin(), angle.cos());
        let t = 1.0 - c;
        [
            t * x * x + c,
            t * x * y - s * z,
            t * x * z + s * y,
            t * x * y + s * z,
            t * y * y + c,
            t * y * z - s * x,
            t * x * z - s * y,
            t * y * z + s * x,
            t * z * z + c,
        ]
    }

    /// The table the exporter writes and the biped names in `flow` are the same
    /// vocabulary; a drift between them would silently mislabel an export.
    #[test]
    fn the_biped_table_matches_the_exporter() {
        let mut resolved = Vec::new();
        for joint in 0..BONE_COUNT {
            let written = crate::flow::biped_name(joint);
            let bone = from_export_name(&written)
                .unwrap_or_else(|| panic!("{written} did not resolve"));
            assert_eq!(BIPED[bone], written, "{written} is out of the table");
            resolved.push(bone);
        }
        resolved.sort();
        resolved.dedup();
        assert_eq!(
            resolved.len(),
            BONE_COUNT,
            "the exporter's names must cover the canonical skeleton once"
        );
        let mut unique: Vec<&str> = BIPED.iter().copied().collect();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), BONE_COUNT, "biped names must be unique");
    }

    /// Character Studio numbers the fingers thumb, index, middle, ring, pinky
    /// and the segments `""`, `1`, `2`; the parser has to undo that exactly.
    #[test]
    fn finger_names_parse_back_to_their_bone() {
        let cases = [
            ("LeftFinger0", "LeftHandThumb1"),
            ("LeftFinger01", "LeftHandThumb2"),
            ("LeftFinger02", "LeftHandThumb3"),
            ("LeftFinger1", "LeftHandIndex1"),
            ("LeftFinger11", "LeftHandIndex2"),
            ("LeftFinger12", "LeftHandIndex3"),
            ("LeftFinger2", "LeftHandMiddle1"),
            ("LeftFinger22", "LeftHandMiddle3"),
            ("LeftFinger3", "LeftHandRing1"),
            ("LeftFinger4", "LeftHandPinky1"),
            ("LeftFinger41", "LeftHandPinky2"),
            ("RightFinger02", "RightHandThumb3"),
            ("RightFinger12", "RightHandIndex3"),
        ];
        for (raw, expected) in cases {
            assert_eq!(
                from_export_name(raw).map(name),
                Some(expected),
                "{raw} did not resolve to {expected}"
            );
        }
        assert_eq!(from_export_name("LeftFinger5"), None);
        assert_eq!(from_export_name("LeftFinger13"), None);
    }

    /// Every vendored name has to resolve back to the bone it was written for:
    /// that is what makes the file usable by the package's own retargeter.
    /// The biped spelling is the exception the table documents — the package
    /// reads a bare `chest` as `Spine1`, ours is the first spine bone — so it is
    /// checked through our own table instead, and the difference is pinned here
    /// so a change in the package's alias table shows up as a failure.
    #[test]
    fn every_rig_name_round_trips() {
        for rig in Rig::ALL {
            for bone in 0..BONE_COUNT {
                let written = rig.name(bone);
                if rig == Rig::Biped {
                    assert_eq!(
                        from_export_name(&written),
                        Some(bone),
                        "biped wrote {written} for {}",
                        BONES[bone]
                    );
                } else {
                    assert_eq!(
                        canonicalize(&written),
                        Some(bone),
                        "{:?} wrote {written} for {}",
                        rig,
                        BONES[bone]
                    );
                }
            }
        }
        assert_eq!(canonicalize("Chest").map(name), Some("Spine1"));
        assert_eq!(from_export_name("Chest").map(name), Some("Spine"));
        assert_eq!(canonicalize("Chest2"), None);
    }

    /// The two places our table deliberately differs from the package: Unreal's
    /// spine (its dedupe fallback answers `Spine` for all three bones) and MMD's
    /// third spine (the standard rig has no name for it).
    #[test]
    fn the_local_additions_name_the_spine_bones() {
        assert_eq!(canonicalize("spine_01").map(name), Some("Spine"));
        assert_eq!(canonicalize("spine_02").map(name), Some("Spine1"));
        assert_eq!(canonicalize("spine_03").map(name), Some("Spine2"));
        assert_eq!(canonicalize("上半身3").map(name), Some("Spine2"));
        // Without the entries the fallback would strip the index and lose the
        // chest, which is what the package does; the standard MMD chain stops at
        // 上半身2, which is Spine1.
        assert_eq!(canonicalize("neck_01").map(name), Some("Neck"));
        assert_eq!(canonicalize("上半身2").map(name), Some("Spine1"));
        for (alias, bone) in EXTRA {
            assert!((*bone as usize) < BONE_COUNT);
            assert!(
                !ALIASES.iter().any(|(known, _)| known == alias),
                "{alias} is already in the package table"
            );
        }
    }

    /// The composition order only works if a parent is always earlier than its
    /// child, which is also what lets `world_rest` do a single pass.
    #[test]
    fn every_parent_precedes_its_child() {
        for (bone, parent) in PARENTS.iter().enumerate() {
            if let Some(parent) = parent {
                assert!(
                    (*parent as usize) < bone,
                    "{} comes before its parent {}",
                    BONES[bone],
                    BONES[*parent as usize]
                );
            }
        }
    }

    /// `world_rest` has to reproduce the package's `CANONICAL_REST_WORLD`; it
    /// is composed from `REST` + `PARENTS` rather than copied, so this pins the
    /// composition, the quaternion convention and the multiplication order
    /// against the package's own numbers.
    #[test]
    fn the_world_rest_matches_the_package() {
        let world = world_rest();
        let close = |actual: &[f64; 9], expected: &[f64; 9], what: &str| {
            for i in 0..9 {
                assert!(
                    (actual[i] - expected[i]).abs() < 1e-6,
                    "{what} entry {i}: {} != {}",
                    actual[i],
                    expected[i]
                );
            }
        };
        // `Hips: [0.0010741149, 6.5e-8, 1e-10, 0.9999994231]`
        close(
            &world[0],
            &quat_matrix(&[0.0010741149, 0.0000000654, 0.0000000001, 0.9999994231]),
            "Hips",
        );
        // `LeftArm` in world space; its local rest is nearly the identity, so
        // the T-pose lives in the shoulder above it.
        let arm_world = quat_matrix(&[0.4889318421, 0.5108287296, -0.5108290885, 0.4889307819]);
        close(&world[7], &arm_world, "LeftArm");
        // Composing the package's own shoulder world with the arm's local rest
        // has to land on the arm's world rest: that is the exact convention
        // `world_rest` uses, checked against data it did not produce.
        let shoulder = quat_matrix(&[0.4605450938, 0.4559956590, -0.5375590068, 0.5394409088]);
        let local_arm = quat_matrix(&[-0.0030895733, 0.0801814846, -0.0250418297, 0.9964608827]);
        close(&matrix_multiply(&shoulder, &local_arm), &arm_world, "arm chain");
    }

    /// Our 52 biped names over the SMPL-H hierarchy *are* the canonical tree:
    /// same root, same 52 bones, every child under its canonical parent.
    #[test]
    fn the_smpl_hierarchy_is_the_canonical_one() {
        let prepared = fixture(identity_motion);
        let canonical: Vec<usize> = prepared
            .names
            .iter()
            .map(|name| from_export_name(name).expect("resolves"))
            .collect();
        let mut seen = canonical.clone();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), BONE_COUNT, "every bone exactly once");
        let mut joint_of = [usize::MAX; BONE_COUNT];
        for (joint, bone) in canonical.iter().enumerate() {
            joint_of[*bone] = joint;
        }
        for (joint, bone) in canonical.iter().enumerate() {
            let expected = PARENTS[*bone]
                .map(|parent| joint_of[parent as usize] as i32)
                .unwrap_or(-1);
            assert_eq!(
                prepared.parents[joint], expected,
                "{} hangs off the wrong bone",
                BONES[*bone]
            );
        }
    }

    /// The default export must not move: `Biped` with no correction keeps the
    /// names and the rotations of the payload it was handed.
    #[test]
    fn the_default_rig_is_a_no_op() {
        let mut prepared = fixture(turned);
        let names = prepared.names.clone();
        let local = prepared.local.clone();
        let coverage = rewrite(&mut prepared, Rig::Biped, false).expect("rewrite");
        assert_eq!(prepared.names, names);
        assert_eq!(prepared.local, local, "the default must not touch rotations");
        assert_eq!(prepared.parents, SMPL_PARENTS.to_vec());
        assert_eq!(coverage, 1.0);
    }

    /// At rest the corrected channels are the canonical local rest, so a rest
    /// pose stays a rest pose on the target rig. The tables are `f32`, so the
    /// composed chain carries a few times 1e-6 of rounding.
    #[test]
    fn a_rest_pose_corrects_to_the_canonical_rest() {
        let mut prepared = fixture(identity_motion);
        rewrite(&mut prepared, Rig::Mixamo, true).expect("rewrite");
        let canonical: Vec<usize> = prepared
            .names
            .iter()
            .map(|name| canonicalize(name).expect("resolves"))
            .collect();
        for (joint, bone) in canonical.iter().enumerate() {
            let rest = quat_matrix(&REST[*bone]);
            for i in 0..9 {
                assert!(
                    (prepared.local[joint * 9 + i] - rest[i]).abs() < 1e-5,
                    "{} rest entry {i}: {} != {}",
                    BONES[*bone],
                    prepared.local[joint * 9 + i],
                    rest[i]
                );
            }
        }
    }

    /// The chain end to end: a clip written by the real BVH writer carries the
    /// vendor's names in its `ROOT`/`JOINT` lines, and every one of them resolves
    /// back through the canonicaliser — which is what a DCC (and the package's
    /// own retargeter) will do with the file. The writer emits the hierarchy
    /// depth-first, so the names are compared as a set, and the root anchors each
    /// rig.
    #[test]
    fn a_written_bvh_carries_the_vendor_names() {
        for rig in [Rig::Mixamo, Rig::Vrm, Rig::Unreal, Rig::Daz, Rig::Mmd] {
            let mut prepared = fixture_frames(2, |frame, joint| turned(joint + frame * 3));
            rewrite(&mut prepared, rig, true).expect("rewrite");
            let path = std::env::temp_dir().join(format!(
                "velomocap-retarget-{}.bvh",
                rig.slug()
            ));
            crate::flow::write_prepared(&path, &prepared)
                .unwrap_or_else(|error| panic!("{:?}: {error:#}", rig));
            let text = std::fs::read_to_string(&path).expect("read back");
            let mut joints: Vec<&str> = text
                .lines()
                .filter(|line| {
                    let line = line.trim_start();
                    line.starts_with("ROOT ") || line.starts_with("JOINT ")
                })
                .map(|line| {
                    line.split_whitespace()
                        .nth(1)
                        .expect("a name after ROOT/JOINT")
                })
                .collect();
            assert_eq!(joints.len(), BONE_COUNT, "{:?}: joint count", rig);
            assert_eq!(joints[0], rig.name(0), "{:?}: the root bone", rig);
            for written in &joints {
                assert!(
                    canonicalize(written).is_some(),
                    "{rig:?} wrote {written}, which does not resolve"
                );
            }
            let mut written = joints.clone();
            written.sort_unstable();
            joints = prepared.names.iter().map(String::as_str).collect();
            joints.sort_unstable();
            assert_eq!(written, joints, "{:?}: the file's bone names", rig);
            let _ = std::fs::remove_file(&path);
        }
    }

    /// The FBX side of the same chain. It used to insist the root bone was
    /// literally called `Hips`, which would have refused every vendored
    /// export — `pelvis`, `hip`, `J_Bip_C_Hips` and `mixamorig:Hips` are all
    /// legitimate roots. MMD's Japanese names stay refused: the FBX writer keeps
    /// bone names ASCII, which is why the interface sends MMD to the BVH.
    #[test]
    fn the_fbx_writer_takes_every_ascii_rig_but_not_mmd() {
        let path = std::env::temp_dir().join("velomocap-retarget.fbx");
        for rig in Rig::ALL {
            if !rig.ascii_names() {
                continue;
            }
            let mut prepared = fixture_frames(2, |frame, joint| turned(joint + frame));
            rewrite(&mut prepared, rig, false).expect("rewrite");
            let report = crate::flow::write_prepared_fbx(&path, &prepared)
                .unwrap_or_else(|error| panic!("{rig:?} was refused: {error:#}"));
            assert_eq!(report.bones, BONE_COUNT, "{rig:?}: bone count");
        }
        let mut mmd = fixture(identity_motion);
        rewrite(&mut mmd, Rig::Mmd, false).expect("rewrite");
        assert!(
            crate::flow::write_prepared_fbx(&path, &mmd).is_err(),
            "the FBX writer is ASCII-only, so MMD has to be refused"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The property the correction exists for: after it, a canonical-rest rig
    /// holds every bone at the world orientation this clip specifies, relative
    /// to its own rest (`world_canonical · restᵀ == world_ours`).
    #[test]
    fn the_correction_preserves_the_world_orientation() {
        let mut ours = fixture(turned);
        let source_world = worlds(&ours.names, &ours.parents, &ours.local, 0);
        // Before the rewrite these are our own biped names, so they resolve
        // through `BIPED` (`LeftShoulder` is the arm here, not the clavicle).
        let canonical: Vec<usize> = ours
            .names
            .iter()
            .map(|name| from_export_name(name).expect("resolves"))
            .collect();
        rewrite(&mut ours, Rig::Vrm, true).expect("rewrite");

        // The same composition, but with the target's rest channels: a rig
        // applies the channel as the local rotation, so its world orientation is
        // the product of the channels along the chain.
        let canonical_world = worlds(&ours.names, &ours.parents, &ours.local, 0);
        let rest = world_rest();
        for (joint, bone) in canonical.iter().enumerate() {
            let expected = matrix_multiply(&source_world[joint], &rest[*bone]);
            for i in 0..9 {
                assert!(
                    (canonical_world[joint][i] - expected[i]).abs() < 3e-5,
                    "{} world entry {i}: {} != {}",
                    BONES[*bone],
                    canonical_world[joint][i],
                    expected[i]
                );
            }
        }
    }

    /// World rotation of every joint for one frame: the product of the local
    /// matrices from the root down.
    fn worlds(names: &[String], parents: &[i32], local: &[f64], frame: usize) -> Vec<[f64; 9]> {
        let bones = parents.len();
        assert_eq!(names.len(), bones);
        let mut out = vec![IDENTITY; bones];
        for joint in 0..bones {
            let at = (frame * bones + joint) * 9;
            let m: [f64; 9] = std::array::from_fn(|i| local[at + i]);
            out[joint] = match parents[joint] {
                parent if parent < 0 => m,
                parent => matrix_multiply(&out[parent as usize], &m),
            };
        }
        out
    }

    /// The correction has to walk every frame with the payload's stride; a
    /// stride mistake would leave the clip's tail uncorrected and still pass a
    /// single-frame test.
    #[test]
    fn every_frame_is_corrected() {
        let frames = 4;
        let rest = world_rest();
        let mut prepared = fixture_frames(frames, |frame, joint| turned(joint + frame * 5));
        let canonical: Vec<usize> = prepared
            .names
            .iter()
            .map(|name| from_export_name(name).expect("resolves"))
            .collect();
        let source: Vec<Vec<[f64; 9]>> = (0..frames)
            .map(|frame| worlds(&prepared.names, &prepared.parents, &prepared.local, frame))
            .collect();
        rewrite(&mut prepared, Rig::Daz, true).expect("rewrite");
        for frame in 0..frames {
            let corrected = worlds(&prepared.names, &prepared.parents, &prepared.local, frame);
            for (joint, bone) in canonical.iter().enumerate() {
                let expected = matrix_multiply(&source[frame][joint], &rest[*bone]);
                for i in 0..9 {
                    assert!(
                        (corrected[joint][i] - expected[i]).abs() < 3e-5,
                        "frame {frame} {} entry {i}",
                        BONES[*bone]
                    );
                }
            }
        }
    }

    /// A payload the canonical skeleton cannot name is refused instead of being
    /// half-renamed — the MHR export writes `mhr000`-style bones.
    #[test]
    fn a_clip_without_canonical_names_is_refused() {
        let mut prepared = fixture(identity_motion);
        prepared.names = (0..BONE_COUNT).map(|bone| format!("mhr{bone:03}")).collect();
        let error = rewrite(&mut prepared, Rig::Mixamo, false).expect_err("must refuse");
        assert!(
            format!("{error:#}").contains("mhr000"),
            "unhelpful error: {error:#}"
        );
        // And a clip it can name, but with a hierarchy that is not the
        // canonical one, cannot be corrected.
        let mut prepared = fixture(identity_motion);
        prepared.parents[20] = 0;
        let error = rewrite(&mut prepared, Rig::Mixamo, true).expect_err("must refuse");
        assert!(
            format!("{error:#}").contains("hangs off the wrong bone"),
            "unhelpful error: {error:#}"
        );
    }

    /// Grip sanity: rig slugs are what the command line and the interface use.
    #[test]
    fn rig_slugs_round_trip() {
        for rig in Rig::ALL {
            assert_eq!(Rig::from_slug(rig.slug()), Some(rig));
            assert_eq!(Rig::from_slug(&rig.slug().to_uppercase()), Some(rig));
        }
        assert_eq!(Rig::from_slug("nope"), None);
        assert!(Rig::Biped.binds_in_t_pose());
        assert!(!Rig::Unreal.binds_in_t_pose());
        assert!(!Rig::Mmd.binds_in_t_pose(), "a PMX bind is per model");
    }

    /// The fingers, spelled out per vendor: this is the part of the table the
    /// package *generates*, so it is the part that was missing, and every rig has
    /// to name all 30 finger bones distinctly and resolvably.
    #[test]
    fn every_rig_names_the_fingers() {
        // Left index segment 2 and left thumb segment 1, per rig.
        let cases: [(Rig, &str, &str); 6] = [
            (Rig::Biped, "LeftFinger11", "LeftFinger0"),
            (Rig::Mixamo, "LeftHandIndex2", "LeftHandThumb1"),
            (Rig::MixamoPrefixed, "mixamorig:LeftHandIndex2", "mixamorig:LeftHandThumb1"),
            (Rig::Vrm, "J_Bip_L_Index2", "J_Bip_L_Thumb1"),
            (Rig::Unreal, "index_02_l", "thumb_01_l"),
            (Rig::Daz, "lIndex2", "lThumb1"),
        ];
        for (rig, index, thumb) in cases {
            assert_eq!(rig.name(11), index, "{rig:?}: left index 2");
            assert_eq!(rig.name(22), thumb, "{rig:?}: left thumb 1");
        }
        // MMD numbers the thumb from 0 and the rest from 1.
        assert_eq!(Rig::Mmd.name(11), "左人指2");
        assert_eq!(Rig::Mmd.name(22), "左親指0");
        assert_eq!(Rig::Mmd.name(24), "左親指2");
        assert_eq!(Rig::Mmd.name(40), "右薬指3");
        for rig in Rig::ALL {
            let mut names: Vec<String> = (0..BONE_COUNT).map(|bone| rig.name(bone)).collect();
            names.sort();
            names.dedup();
            assert_eq!(names.len(), BONE_COUNT, "{rig:?}: duplicate bone names");
            let fingers = (0..BONE_COUNT)
                .filter(|bone| finger_parts(*bone).is_some())
                .count();
            assert_eq!(fingers, 30, "{rig:?}: finger count");
        }
    }
}
