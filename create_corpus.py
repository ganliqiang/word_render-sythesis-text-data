# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Sqrt5


import sys
sys.path.append('..')
import random
import time
from decimal import Decimal
import math

def progressbar(cur, total, begin_time = 0, cur_time = 0, info = ''):
    sys.stdout.write('\r')
    if begin_time == 0 and cur_time == 0 or cur == 0:
        sys.stdout.write("[%-30s]\t%6.2f%%" %
                         ('=' * int(math.floor(cur * 30 / total)),
                         ('=' * int(math.floor(cur * 30 / total)),
                          float(cur * 100) / total)))
    else:
        sys.stdout.write("[%-30s]  %6.2f%%   COST: %.2fs   ETA: %.2fs   %s     " %
                         ('=' * int(math.floor(cur * 30 / total)),
                          float(cur * 100) / total,
                          (cur_time - begin_time),
                          (cur_time - begin_time) * (total - cur) / cur,info))
    if cur == total:
        print ''
    sys.stdout.flush()

    return

def common_used_character(tranditional = True, radical = True, alphabeta = True, number = True, serial_number = True, symbol = True):
    chars_6721 = u'啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂盎凹敖熬翱袄傲奥懊澳芭捌扒叭吧笆八疤巴拔跋靶把耙' \
                 u'坝霸罢爸白柏百摆佰败拜稗斑班搬扳般颁板版扮拌伴瓣半办绊邦帮梆榜膀绑棒磅蚌镑傍谤苞胞包褒剥薄雹保堡饱宝' \
                 u'抱报暴豹鲍爆杯碑悲卑北辈背贝钡倍狈备惫焙被奔苯本笨崩绷甭泵蹦迸逼鼻比鄙笔彼碧蓖蔽毕毙毖币庇痹闭敝弊必' \
                 u'辟壁臂避陛鞭边编贬扁便变卞辨辩辫遍标彪膘表鳖憋别瘪彬斌濒滨宾摈兵冰柄丙秉饼炳病并玻菠播拨钵波博勃搏铂' \
                 u'箔伯帛舶脖膊渤泊驳捕卜哺补埠不布步簿部怖擦猜裁材才财睬踩采彩菜蔡餐参蚕残惭惨灿苍舱仓沧藏操糙槽曹草厕' \
                 u'策侧册测层蹭插叉茬茶查碴搽察岔差诧拆柴豺搀掺蝉馋谗缠铲产阐颤昌猖场尝常长偿肠厂敞畅唱倡超抄钞朝嘲潮巢' \
                 u'吵炒车扯撤掣彻澈郴臣辰尘晨忱沉陈趁衬撑称城橙成呈乘程惩澄诚承逞骋秤吃痴持匙池迟弛驰耻齿侈尺赤翅斥炽充' \
                 u'冲虫崇宠抽酬畴踌稠愁筹仇绸瞅丑臭初出橱厨躇锄雏滁除楚础储矗搐触处揣川穿椽传船喘串疮窗幢床闯创吹炊捶锤' \
                 u'垂春椿醇唇淳纯蠢戳绰疵茨磁雌辞慈瓷词此刺赐次聪葱囱匆从丛凑粗醋簇促蹿篡窜摧崔催脆瘁粹淬翠村存寸磋撮搓' \
                 u'措挫错搭达答瘩打大呆歹傣戴带殆代贷袋待逮怠耽担丹单郸掸胆旦氮但惮淡诞弹蛋当挡党荡档刀捣蹈倒岛祷导到稻' \
                 u'悼道盗德得的蹬灯登等瞪凳邓堤低滴迪敌笛狄涤翟嫡抵底地蒂第帝弟递缔颠掂滇碘点典靛垫电佃甸店惦奠淀殿碉叼' \
                 u'雕凋刁掉吊钓调跌爹碟蝶迭谍叠丁盯叮钉顶鼎锭定订丢东冬董懂动栋侗恫冻洞兜抖斗陡豆逗痘都督毒犊独读堵睹赌' \
                 u'杜镀肚度渡妒端短锻段断缎堆兑队对墩吨蹲敦顿囤钝盾遁掇哆多夺垛躲朵跺舵剁惰堕蛾峨鹅俄额讹娥恶厄扼遏鄂饿' \
                 u'恩而儿耳尔饵洱二贰发罚筏伐乏阀法珐藩帆番翻樊矾钒繁凡烦反返范贩犯饭泛坊芳方肪房防妨仿访纺放菲非啡飞肥' \
                 u'匪诽吠肺废沸费芬酚吩氛分纷坟焚汾粉奋份忿愤粪丰封枫蜂峰锋风疯烽逢冯缝讽奉凤佛否夫敷肤孵扶拂辐幅氟符伏' \
                 u'俘服浮涪福袱弗甫抚辅俯釜斧脯腑府腐赴副覆赋复傅付阜父腹负富讣附妇缚咐噶嘎该改概钙盖溉干甘杆柑竿肝赶感' \
                 u'秆敢赣冈刚钢缸肛纲岗港杠篙皋高膏羔糕搞镐稿告哥歌搁戈鸽胳疙割革葛格蛤阁隔铬个各给根跟耕更庚羹埂耿梗工' \
                 u'攻功恭龚供躬公宫弓巩汞拱贡共钩勾沟苟狗垢构购够辜菇咕箍估沽孤姑鼓古蛊骨谷股故顾固雇刮瓜剐寡挂褂乖拐怪' \
                 u'棺关官冠观管馆罐惯灌贯光广逛瑰规圭硅归龟闺轨鬼诡癸桂柜跪贵刽辊滚棍锅郭国果裹过哈骸孩海氦亥害骇酣憨邯' \
                 u'韩含涵寒函喊罕翰撼捍旱憾悍焊汗汉夯杭航壕嚎豪毫郝好耗号浩呵喝荷菏核禾和何合盒貉阂河涸赫褐鹤贺嘿黑痕很' \
                 u'狠恨哼亨横衡恒轰哄烘虹鸿洪宏弘红喉侯猴吼厚候后呼乎忽瑚壶葫胡蝴狐糊湖弧虎唬护互沪户花哗华猾滑画划化话' \
                 u'槐徊怀淮坏欢环桓还缓换患唤痪豢焕涣宦幻荒慌黄磺蝗簧皇凰惶煌晃幌恍谎灰挥辉徽恢蛔回毁悔慧卉惠晦贿秽会烩' \
                 u'汇讳诲绘荤昏婚魂浑混豁活伙火获或惑霍货祸击圾基机畸稽积箕肌饥迹激讥鸡姬绩缉吉极棘辑籍集及急疾汲即嫉级' \
                 u'挤几脊己蓟技冀季伎祭剂悸济寄寂计记既忌际妓继纪嘉枷夹佳家加荚颊贾甲钾假稼价架驾嫁歼监坚尖笺间煎兼肩艰' \
                 u'奸缄茧检柬碱硷拣捡简俭剪减荐槛鉴践贱见键箭件健舰剑饯渐溅涧建僵姜将浆江疆蒋桨奖讲匠酱降蕉椒礁焦胶交郊' \
                 u'浇骄娇嚼搅铰矫侥脚狡角饺缴绞剿教酵轿较叫窖揭接皆秸街阶截劫节桔杰捷睫竭洁结解姐戒藉芥界借介疥诫届巾筋' \
                 u'斤金今津襟紧锦仅谨进靳晋禁近烬浸尽劲荆兢茎睛晶鲸京惊精粳经井警景颈静境敬镜径痉靖竟竞净炯窘揪究纠玖韭' \
                 u'久灸九酒厩救旧臼舅咎就疚鞠拘狙疽居驹菊局咀矩举沮聚拒据巨具距踞锯俱句惧炬剧捐鹃娟倦眷卷绢撅攫抉掘倔爵' \
                 u'觉决诀绝均菌钧军君峻俊竣浚郡骏喀咖卡咯开揩楷凯慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕颗科壳咳' \
                 u'可渴克刻客课肯啃垦恳坑吭空恐孔控抠口扣寇枯哭窟苦酷库裤夸垮挎跨胯块筷侩快宽款匡筐狂框矿眶旷况亏盔岿窥' \
                 u'葵奎魁傀馈愧溃坤昆捆困括扩廓阔垃拉喇蜡腊辣啦莱来赖蓝婪栏拦篮阑兰澜谰揽览懒缆烂滥琅榔狼廊郎朗浪捞劳牢' \
                 u'老佬姥酪烙涝勒乐雷镭蕾磊累儡垒擂肋类泪棱楞冷厘梨犁黎篱狸离漓理李里鲤礼莉荔吏栗丽厉励砾历利傈例俐痢立' \
                 u'粒沥隶力璃哩俩联莲连镰廉怜涟帘敛脸链恋炼练粮凉梁粱良两辆量晾亮谅撩聊僚疗燎寥辽潦了撂镣廖料列裂烈劣猎' \
                 u'琳林磷霖临邻鳞淋凛赁吝拎玲菱零龄铃伶羚凌灵陵岭领另令溜琉榴硫馏留刘瘤流柳六龙聋咙笼窿隆垄拢陇楼娄搂篓' \
                 u'漏陋芦卢颅庐炉掳卤虏鲁麓碌露路赂鹿潞禄录陆戮驴吕铝侣旅履屡缕虑氯律率滤绿峦挛孪滦卵乱掠略抡轮伦仑沦纶' \
                 u'论萝螺罗逻锣箩骡裸落洛骆络妈麻玛码蚂马骂嘛吗埋买麦卖迈脉瞒馒蛮满蔓曼慢漫谩芒茫盲氓忙莽猫茅锚毛矛铆卯' \
                 u'茂冒帽貌贸么玫枚梅酶霉煤没眉媒镁每美昧寐妹媚门闷们萌蒙檬盟锰猛梦孟眯醚靡糜迷谜弥米秘觅泌蜜密幂棉眠绵' \
                 u'冕免勉娩缅面苗描瞄藐秒渺庙妙蔑灭民抿皿敏悯闽明螟鸣铭名命谬摸摹蘑模膜磨摩魔抹末莫墨默沫漠寞陌谋牟某拇' \
                 u'牡亩姆母墓暮幕募慕木目睦牧穆拿哪呐钠那娜纳氖乃奶耐奈南男难囊挠脑恼闹淖呢馁内嫩能妮霓倪泥尼拟你匿腻逆' \
                 u'溺蔫拈年碾撵捻念娘酿鸟尿捏聂孽啮镊镍涅您柠狞凝宁拧泞牛扭钮纽脓浓农弄奴努怒女暖虐疟挪懦糯诺哦欧鸥殴藕' \
                 u'呕偶沤啪趴爬帕怕琶拍排牌徘湃派攀潘盘磐盼畔判叛乓庞旁耪胖抛咆刨炮袍跑泡呸胚培裴赔陪配佩沛喷盆砰抨烹澎' \
                 u'彭蓬棚硼篷膨朋鹏捧碰坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏片骗飘漂瓢票撇瞥拼频贫品聘乒坪苹萍平凭瓶评' \
                 u'屏坡泼颇婆破魄迫粕剖扑铺仆莆葡菩蒲埔朴圃普浦谱曝瀑期欺栖戚妻七凄漆柒沏其棋奇歧畦崎脐齐旗祈祁骑起岂乞' \
                 u'企启契砌器气迄弃汽泣讫掐恰洽牵扦钎铅千迁签仟谦乾黔钱钳前潜遣浅谴堑嵌欠歉枪呛腔羌墙蔷强抢橇锹敲悄桥瞧' \
                 u'乔侨巧鞘撬翘峭俏窍切茄且怯窃钦侵亲秦琴勤芹擒禽寝沁青轻氢倾卿清擎晴氰情顷请庆琼穷秋丘邱球求囚酋泅趋区' \
                 u'蛆曲躯屈驱渠取娶龋趣去圈颧权醛泉全痊拳犬券劝缺炔瘸却鹊榷确雀裙群然燃冉染瓤壤攘嚷让饶扰绕惹热壬仁人忍' \
                 u'韧任认刃妊纫扔仍日戎茸蓉荣融熔溶容绒冗揉柔肉茹蠕儒孺如辱乳汝入褥软阮蕊瑞锐闰润若弱撒洒萨腮鳃塞赛三叁' \
                 u'伞散桑嗓丧搔骚扫嫂瑟色涩森僧莎砂杀刹沙纱傻啥煞筛晒珊苫杉山删煽衫闪陕擅赡膳善汕扇缮墒伤商赏晌上尚裳梢' \
                 u'捎稍烧芍勺韶少哨邵绍奢赊蛇舌舍赦摄射慑涉社设砷申呻伸身深娠绅神沈审婶甚肾慎渗声生甥牲升绳省盛剩胜圣师' \
                 u'失狮施湿诗尸虱十石拾时什食蚀实识史矢使屎驶始式示士世柿事拭誓逝势是嗜噬适仕侍释饰氏市恃室视试收手首守' \
                 u'寿授售受瘦兽蔬枢梳殊抒输叔舒淑疏书赎孰熟薯暑曙署蜀黍鼠属术述树束戍竖墅庶数漱恕刷耍摔衰甩帅栓拴霜双爽' \
                 u'谁水睡税吮瞬顺舜说硕朔烁斯撕嘶思私司丝死肆寺嗣四伺似饲巳松耸怂颂送宋讼诵搜艘擞嗽苏酥俗素速粟僳塑溯宿' \
                 u'诉肃酸蒜算虽隋随绥髓碎岁穗遂隧祟孙损笋蓑梭唆缩琐索锁所塌他它她塔獭挞蹋踏胎苔抬台泰酞太态汰坍摊贪瘫滩' \
                 u'坛檀痰潭谭谈坦毯袒碳探叹炭汤塘搪堂棠膛唐糖倘躺淌趟烫掏涛滔绦萄桃逃淘陶讨套特藤腾疼誊梯剔踢锑提题蹄啼' \
                 u'体替嚏惕涕剃屉天添填田甜恬舔腆挑条迢眺跳贴铁帖厅听烃汀廷停亭庭挺艇通桐酮瞳同铜彤童桶捅筒统痛偷投头透' \
                 u'凸秃突图徒途涂屠土吐兔湍团推颓腿蜕褪退吞屯臀拖托脱鸵陀驮驼椭妥拓唾挖哇蛙洼娃瓦袜歪外豌弯湾玩顽丸烷完' \
                 u'碗挽晚皖惋宛婉万腕汪王亡枉网往旺望忘妄威巍微危韦违桅围唯惟为潍维苇萎委伟伪尾纬未蔚味畏胃喂魏位渭谓尉' \
                 u'慰卫瘟温蚊文闻纹吻稳紊问嗡翁瓮挝蜗涡窝我斡卧握沃巫呜钨乌污诬屋无芜梧吾吴毋武五捂午舞伍侮坞戊雾晤物勿' \
                 u'务悟误昔熙析西硒矽晰嘻吸锡牺稀息希悉膝夕惜熄烯溪汐犀檄袭席习媳喜铣洗系隙戏细瞎虾匣霞辖暇峡侠狭下厦夏' \
                 u'吓掀锨先仙鲜纤咸贤衔舷闲涎弦嫌显险现献县腺馅羡宪陷限线相厢镶香箱襄湘乡翔祥详想响享项巷橡像向象萧硝霄' \
                 u'削哮嚣销消宵淆晓小孝校肖啸笑效楔些歇蝎鞋协挟携邪斜胁谐写械卸蟹懈泄泻谢屑薪芯锌欣辛新忻心信衅星腥猩惺' \
                 u'兴刑型形邢行醒幸杏性姓兄凶胸匈汹雄熊休修羞朽嗅锈秀袖绣墟戌需虚嘘须徐许蓄酗叙旭序畜恤絮婿绪续轩喧宣悬' \
                 u'旋玄选癣眩绚靴薛学穴雪血勋熏循旬询寻驯巡殉汛训讯逊迅压押鸦鸭呀丫芽牙蚜崖衙涯雅哑亚讶焉咽阉烟淹盐严研' \
                 u'蜒岩延言颜阎炎沿奄掩眼衍演艳堰燕厌砚雁唁彦焰宴谚验殃央鸯秧杨扬佯疡羊洋阳氧仰痒养样漾邀腰妖瑶摇尧遥窑' \
                 u'谣姚咬舀药要耀椰噎耶爷野冶也页掖业叶曳腋夜液一壹医揖铱依伊衣颐夷遗移仪胰疑沂宜姨彝椅蚁倚已乙矣以艺抑' \
                 u'易邑屹亿役臆逸肄疫亦裔意毅忆义益溢诣议谊译异翼翌绎茵荫因殷音阴姻吟银淫寅饮尹引隐印英樱婴鹰应缨莹萤营' \
                 u'荧蝇迎赢盈影颖硬映哟拥佣臃痈庸雍踊蛹咏泳涌永恿勇用幽优悠忧尤由邮铀犹油游酉有友右佑釉诱又幼迂淤于盂榆' \
                 u'虞愚舆余俞逾鱼愉渝渔隅予娱雨与屿禹宇语羽玉域芋郁吁遇喻峪御愈欲狱育誉浴寓裕预豫驭鸳渊冤元垣袁原援辕园' \
                 u'员圆猿源缘远苑愿怨院曰约越跃钥岳粤月悦阅耘云郧匀陨允运蕴酝晕韵孕匝砸杂栽哉灾宰载再在咱攒暂赞赃脏葬遭' \
                 u'糟凿藻枣早澡蚤躁噪造皂灶燥责择则泽贼怎增憎曾赠扎喳渣札轧铡闸眨栅榨咋乍炸诈摘斋宅窄债寨瞻毡詹粘沾盏斩' \
                 u'辗崭展蘸栈占战站湛绽樟章彰漳张掌涨杖丈帐账仗胀瘴障招昭找沼赵照罩兆肇召遮折哲蛰辙者锗蔗这浙珍斟真甄砧' \
                 u'臻贞针侦枕疹诊震振镇阵蒸挣睁征狰争怔整拯正政帧症郑证芝枝支吱蜘知肢脂汁之织职直植殖执值侄址指止趾只旨' \
                 u'纸志挚掷至致置帜峙制智秩稚质炙痔滞治窒中盅忠钟衷终种肿重仲众舟周州洲诌粥轴肘帚咒皱宙昼骤珠株蛛朱猪诸' \
                 u'诛逐竹烛煮拄瞩嘱主著柱助蛀贮铸筑住注祝驻抓爪拽专砖转撰赚篆桩庄装妆撞壮状椎锥追赘坠缀谆准捉拙卓桌琢茁' \
                 u'酌啄着灼浊兹咨资姿滋淄孜紫仔籽滓子自渍字鬃棕踪宗综总纵邹走奏揍租足卒族祖诅阻组钻纂嘴醉最罪尊遵昨左佐' \
                 u'柞做作坐座亍丌兀丐廿卅丕亘丞鬲孬噩禺匕乇夭爻卮氐囟胤馗毓睾鼗亟鼐乜乩亓芈孛啬嘏仄厍厝厣厥厮靥赝叵匦匮' \
                 u'匾赜卦卣刈刎刭刳刿剀剌剞剡剜蒯剽劂劁劐劓罔仃仉仂仨仡仫仞伛仳伢佤仵伥伧伉伫佞佧攸佚佝佟佗伲伽佶佴侑侉' \
                 u'侃侏佾佻侪佼侬侔俦俨俪俅俚俣俜俑俟俸倩偌俳倬倏倮倭俾倜倌倥倨偾偃偕偈偎偬偻傥傧傩傺僖儆僭僬僦僮儇儋仝' \
                 u'氽佘佥俎龠汆籴兮巽黉馘冁夔匍訇匐凫夙兕兖亳衮袤亵脔裒禀嬴蠃羸冱冽冼凇冢冥讦讧讪讴讵讷诂诃诋诏诎诒诓诔' \
                 u'诖诘诙诜诟诠诤诨诩诮诰诳诶诹诼诿谀谂谄谇谌谏谑谒谔谕谖谙谛谘谝谟谠谡谥谧谪谫谮谯谲谳谵谶卺阢阡阱阪阽' \
                 u'阼陂陉陔陟陧陬陲陴隈隍隗隰邗邛邝邙邬邡邴邳邶邺邸邰郏郅邾郐郄郇郓郦郢郜郗郛郫郯郾鄄鄢鄞鄣鄱鄯鄹酃酆刍' \
                 u'奂劢劬劭劾哿勐勖勰叟燮矍凼鬯厶弁畚巯坌垩垡塾墼壅壑圩圬圪圳圹圮圯坜圻坂坩垅坫垆坼坻坨坭坶坳垭垤垌垲埏' \
                 u'垧垴垓垠埕埘埚埙埒垸埴埯埸埤埝堋堍埽埭堀堞堙塄堠塥塬墁墉墚墀馨鼙懿艽艿芏芊芨芄芎芑芗芙芫芸芾芰苈苊苣' \
                 u'芘芷芮苋苌苁芩芴芡芪芟苄苎芤苡茉苷苤茏茇苜苴苒苘茌苻苓茑茚茆茔茕苠苕茜荑荛荜茈莒茼茴茱莛荞茯荏荇荃荟' \
                 u'荀茗荠茭茺茳荦荥荨茛荩荬荪荭荮莰荸莳莴莠莪莓莜莅荼莶莩荽莸荻莘莞莨莺莼菁萁菥菘堇萘萋菝菽菖萜萸萑萆菔' \
                 u'菟萏萃菸菹菪菅菀萦菰菡葜葑葚葙葳蒇蒈葺蒉葸萼葆葩葶蒌蒎萱葭蓁蓍蓐蓦蒽蓓蓊蒿蒺蓠蒡蒹蒴蒗蓥蓣蔌甍蔸蓰蔹' \
                 u'蔟蔺蕖蔻蓿蓼蕙蕈蕨蕤蕞蕺瞢蕃蕲蕻薤薨薇薏蕹薮薜薅薹薷薰藓藁藜藿蘧蘅蘩蘖蘼弈夼奁耷奕奚奘匏尢尥尬尴扪抟' \
                 u'抻拊拚拗拮挢拶挹捋捃掭揶捱捺掎掴捭掬掊捩掮掼揲揸揠揿揄揞揎摒揆掾摅摁搋搛搠搌搦搡摞撄摭撖摺撷撸撙撺擀' \
                 u'擐擗擤擢攉攥攮弋忒甙弑卟叱叽叩叨叻吒吖吆呋呒呓呔呖呃吡呗呙吣吲咂咔呷呱呤咚咛咄呶呦咝哐咭哂咴哒咧咦哓' \
                 u'哔呲咣哕咻咿哌哙哚哜咩咪咤哝哏哞唛哧唠哽唔哳唢唣唏唑唧唪啧喏喵啉啭啁啕唿啐唼唷啖啵啶啷唳唰啜喋嗒喃喱' \
                 u'喹喈喁喟啾嗖喑啻嗟喽喾喔喙嗪嗷嗉嘟嗑嗫嗬嗔嗦嗝嗄嗯嗥嗲嗳嗌嗍嗨嗵嗤辔嘞嘈嘌嘁嘤嘣嗾嘀嘧嘭噘嘹噗嘬噍噢' \
                 u'噙噜噌噔嚆噤噱噫噻噼嚅嚓嚯囔囝囡囵囫囹囿圄圊圉圜帏帙帔帑帱帻帼帷幄幔幛幞幡岌屺岍岐岖岈岘岙岑岚岜岵岢' \
                 u'岽岬岫岱岣峁岷峄峒峤峋峥崂崃崧崦崮崤崞崆崛嵘崾崴崽嵬嵛嵯嵝嵫嵋嵊嵩嵴嶂嶙嶝豳嶷巅彳彷徂徇徉後徕徙徜徨' \
                 u'徭徵徼衢犰犴犷犸狃狁狎狍狒狨狯狩狲狴狷猁狳猃狺狻猗猓猡猊猞猝猕猢猹猥猬猸猱獐獍獗獠獬獯獾舛夥飧夤饧饨' \
                 u'饩饪饫饬饴饷饽馀馄馇馊馍馐馑馓馔馕庀庑庋庖庥庠庹庵庾庳赓廒廑廛廨廪膺忉忖忏怃忮怄忡忤忾怅怆忪忭忸怙怵' \
                 u'怦怛怏怍怩怫怊怿怡恸恹恻恺恂恪恽悖悚悭悝悃悒悌悛惬悻悱惝惘惆惚悴愠愦愕愣惴愀愎愫慊慵憬憔憧憷懔懵忝隳' \
                 u'闩闫闱闳闵闶闼闾阃阄阆阈阊阋阌阍阏阒阕阖阗阙阚爿戕汔汜汊沣沅沐沔沌汨汩汴汶沆沩泐泔沭泷泸泱泗沲泠泖泺' \
                 u'泫泮沱泓泯泾洹洧洌浃浈洇洄洙洎洫浍洮洵洚浏浒浔洳涑浯涞涠浞涓涔浜浠浼浣渚淇淅淞渎涿淠渑淦淝淙渖涫渌涮' \
                 u'渫湮湎湫溲湟溆湓湔渲渥湄滟溱溘滠漭滢溥溧溽溻溷滗溴滏溏滂溟潢潆潇漤漕滹漯漶潋潴漪漉漩澉澍澌潸潲潼潺濑' \
                 u'濉澧澹澶濂濡濮濞濠濯瀚瀣瀛瀹瀵灏灞宄宕宓宥宸甯骞搴寤寮褰寰蹇謇迓迕迥迮迤迩迦迳迨逅逄逋逦逑逍逖逡逵逶' \
                 u'逭逯遄遑遒遐遨遘遢遛暹遴遽邂邈邃邋彗彖彘尻咫屐屙孱屣屦羼弪弩弭艴弼鬻妁妃妍妩妪妣妗姊妫妞妤姒妲妯姗妾' \
                 u'娅娆姝娈姣姘姹娌娉娲娴娑娣娓婀婧婊婕娼婢婵胬媪媛婷婺媾嫫媲嫒嫔媸嫠嫣嫱嫖嫦嫘嫜嬉嬗嬖嬲嬷孀尕尜孚孥孳' \
                 u'孑孓孢驵驷驸驺驿驽骀骁骅骈骊骐骒骓骖骘骛骜骝骟骠骢骣骥骧纡纣纥纨纩纭纰纾绀绁绂绉绋绌绐绔绗绛绠绡绨绫' \
                 u'绮绯绱绲缍绶绺绻绾缁缂缃缇缈缋缌缏缑缒缗缙缜缛缟缡缢缣缤缥缦缧缪缫缬缭缯缰缱缲缳缵幺畿甾邕玎玑玮玢玟' \
                 u'珏珂珑玷玳珀珉珈珥珙顼琊珩珧珞玺珲琏琪瑛琦琥琨琰琮琬琛琚瑁瑜瑗瑕瑙瑷瑭瑾璜璎璀璁璇璋璞璨璩璐璧瓒璺韪' \
                 u'韫韬杌杓杞杈杩枥枇杪杳枘枧杵枨枞枭枋杷杼柰栉柘栊柩枰栌柙枵柚枳柝栀柃枸柢栎柁柽栲栳桠桡桎桢桄桤梃栝桕' \
                 u'桦桁桧桀栾桊桉栩梵梏桴桷梓桫棂楮棼椟椠棹椤棰椋椁楗棣椐楱椹楠楂楝榄楫榀榘楸椴槌榇榈槎榉楦楣楹榛榧榻榫' \
                 u'榭槔榱槁槊槟榕槠榍槿樯槭樗樘橥槲橄樾檠橐橛樵檎橹樽樨橘橼檑檐檩檗檫猷獒殁殂殇殄殒殓殍殚殛殡殪轫轭轱轲' \
                 u'轳轵轶轸轷轹轺轼轾辁辂辄辇辋辍辎辏辘辚軎戋戗戛戟戢戡戥戤戬臧瓯瓴瓿甏甑甓旮旯旰昊昙杲昃昕昀炅曷昝昴昱' \
                 u'昶昵耆晟晔晁晏晖晡晗晷暄暌暧暝暾曛曜曦曩贲贳贶贻贽赀赅赆赈赉赇赍赕赙觇觊觋觌觎觏觐觑牮犟牝牦牯牾牿犄' \
                 u'犋犍犏犒挈挲掰搿擘耄毪毳毽毵毹氅氇氆氍氕氘氙氚氡氩氤氪氲敕敫牍牒牖爰虢刖肟肜肓肼朊肽肱肫肭肴肷胧胨胩' \
                 u'胪胛胂胄胙胍胗朐胝胫胱胴胭脍脎胲胼朕脒豚脶脞脬脘脲腈腌腓腴腙腚腱腠腩腼腽腭腧塍媵膈膂膑滕膣膪臌朦臊膻' \
                 u'臁膦欤欷欹歃歆歙飑飒飓飕飙飚殳彀毂觳斐齑斓於旆旄旃旌旎旒旖炀炜炖炝炻烀炷炫炱烨烊焐焓焖焯焱煳煜煨煅煲' \
                 u'煊煸煺熘熳熵熨熠燠燔燧燹爝爨焘煦熹戾戽扃扈扉祀祆祉祛祜祓祚祢祗祠祯祧祺禅禊禚禧禳忑忐怼恝恚恧恁恙恣悫' \
                 u'愆愍慝憩憝懋懑戆聿沓泶淼矶矸砀砉砗砘砑斫砭砜砝砹砺砻砟砼砥砬砣砩硎硭硖硗砦硐硇硌硪碛碓碚碇碜碡碣碲碹' \
                 u'碥磔磙磉磬磲礅磴礓礤礞礴龛黹黻黼盱眄眍盹眇眈眚眢眙眭眦眵眸睐睑睇睃睚睨睢睥睿瞍睽瞀瞌瞑瞟瞠瞰瞵瞽町畀' \
                 u'畎畋畈畛畲畹疃罘罡罟詈罨罴罱罹羁罾盍盥蠲钆钇钋钊钌钍钏钐钔钗钕钚钛钜钣钤钫钪钭钬钯钰钲钴钶钷钸钹钺钼' \
                 u'钽钿铄铈铉铊铋铌铍铎铐铑铒铕铖铗铙铘铛铞铟铠铢铤铥铧铨铪铩铫铮铯铳铴铵铷铹铼铽铿锃锂锆锇锉锊锍锎锏锒' \
                 u'锓锔锕锖锘锛锝锞锟锢锪锫锩锬锱锲锴锶锷锸锼锾锿镂锵镄镅镆镉镌镎镏镒镓镔镖镗镘镙镛镞镟镝镡镢镤镥镦镧镨' \
                 u'镩镪镫镬镯镱镲镳锺矧矬雉秕秭秣秫稆嵇稃稂稞稔稹稷穑黏馥穰皈皎皓皙皤瓞瓠甬鸠鸢鸨鸩鸪鸫鸬鸲鸱鸶鸸鸷鸹鸺' \
                 u'鸾鹁鹂鹄鹆鹇鹈鹉鹋鹌鹎鹑鹕鹗鹚鹛鹜鹞鹣鹦鹧鹨鹩鹪鹫鹬鹱鹭鹳疔疖疠疝疬疣疳疴疸痄疱疰痃痂痖痍痣痨痦痤痫' \
                 u'痧瘃痱痼痿瘐瘀瘅瘌瘗瘊瘥瘘瘕瘙瘛瘼瘢瘠癀瘭瘰瘿瘵癃瘾瘳癍癞癔癜癖癫癯翊竦穸穹窀窆窈窕窦窠窬窨窭窳衩衲' \
                 u'衽衿袂袢裆袷袼裉裢裎裣裥裱褚裼裨裾裰褡褙褓褛褊褴褫褶襁襦襻疋胥皲皴矜耒耔耖耜耠耢耥耦耧耩耨耱耋耵聃聆' \
                 u'聍聒聩聱覃顸颀颃颉颌颍颏颔颚颛颞颟颡颢颥颦虔虬虮虿虺虼虻蚨蚍蚋蚬蚝蚧蚣蚪蚓蚩蚶蛄蚵蛎蚰蚺蚱蚯蛉蛏蚴蛩' \
                 u'蛱蛲蛭蛳蛐蜓蛞蛴蛟蛘蛑蜃蜇蛸蜈蜊蜍蜉蜣蜻蜞蜥蜮蜚蜾蝈蜴蜱蜩蜷蜿螂蜢蝽蝾蝻蝠蝰蝌蝮螋蝓蝣蝼蝤蝙蝥螓螯螨' \
                 u'蟒蟆螈螅螭螗螃螫蟥螬螵螳蟋蟓螽蟑蟀蟊蟛蟪蟠蟮蠖蠓蟾蠊蠛蠡蠹蠼缶罂罄罅舐竺竽笈笃笄笕笊笫笏筇笸笪笙笮笱' \
                 u'笠笥笤笳笾笞筘筚筅筵筌筝筠筮筻筢筲筱箐箦箧箸箬箝箨箅箪箜箢箫箴篑篁篌篝篚篥篦篪簌篾篼簏簖簋簟簪簦簸籁' \
                 u'籀臾舁舂舄臬衄舡舢舣舭舯舨舫舸舻舳舴舾艄艉艋艏艚艟艨衾袅袈裘裟襞羝羟羧羯羰羲籼敉粑粝粜粞粢粲粼粽糁糇' \
                 u'糌糍糈糅糗糨艮暨羿翎翕翥翡翦翩翮翳糸絷綦綮繇纛麸麴赳趄趔趑趱赧赭豇豉酊酐酎酏酤酢酡酰酩酯酽酾酲酴酹醌' \
                 u'醅醐醍醑醢醣醪醭醮醯醵醴醺豕鹾趸跫踅蹙蹩趵趿趼趺跄跖跗跚跞跎跏跛跆跬跷跸跣跹跻跤踉跽踔踝踟踬踮踣踯踺' \
                 u'蹀踹踵踽踱蹉蹁蹂蹑蹒蹊蹰蹶蹼蹯蹴躅躏躔躐躜躞豸貂貊貅貘貔斛觖觞觚觜觥觫觯訾謦靓雩雳雯霆霁霈霏霎霪霭霰' \
                 u'霾龀龃龅龆龇龈龉龊龌黾鼋鼍隹隼隽雎雒瞿雠銎銮鋈錾鍪鏊鎏鐾鑫鱿鲂鲅鲆鲇鲈稣鲋鲎鲐鲑鲒鲔鲕鲚鲛鲞鲟鲠鲡鲢' \
                 u'鲣鲥鲦鲧鲨鲩鲫鲭鲮鲰鲱鲲鲳鲴鲵鲶鲷鲺鲻鲼鲽鳄鳅鳆鳇鳊鳋鳌鳍鳎鳏鳐鳓鳔鳕鳗鳘鳙鳜鳝鳟鳢靼鞅鞑鞒鞔鞯鞫鞣' \
                 u'鞲鞴骱骰骷鹘骶骺骼髁髀髅髂髋髌髑魅魃魇魉魈魍魑飨餍餮饕饔髟髡髦髯髫髻髭髹鬈鬏鬓鬟鬣麽麾縻麂麇麈麋麒鏖' \
                 u'麝麟黛黜黝黠黟黢黩黧黥黪黯鼢鼬鼯鼹鼷鼽鼾齄'
    chars_42_radical = u'丨丿丶匚刂冂亻勹亠冫冖讠卩阝廴凵艹廾扌囗彡犭夂饣忄丬氵宀辶彐屮纟巛攴攵灬礻肀钅疒衤虍'
    chars_10_number = u'0123456789'
    chars_10_serial_number = u'①②③④⑤⑥⑦⑧⑨⑩'
    chars_26_uppercase = u'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chars_26_lowercase = u'abcdefghijklmnopqrstuvwxyz'
    chars_40_punctuation = u'!"#$%&()*+,-./:;<=>?@[\]^`{|}~、。々…‘’“”《》'
    chars_44_special_symbol = u'±×÷∧∨∑∏∪∩∈√⊥∥⊙∫∮≡≌≈∽∝≠≤≥∞∵∴℃￡‰§№☆○◎◇□△※→←↑↓￥'
    chars_422 = u'乂亶亸伋伾俵俶倓倞倴倻偲僇僰剅剕剟勍勩匜卻厾唝啰啴喆喤嗞嘚嘡噀嚄嚚垍垕垞垟垯垱埇埗埼堉堌堎堲塃塆塮墈' \
                u'墦夬奭姞姽婞婳嫚嫪宬屃峂峃峣峧崟嵎嵖嵚嶓巉帡幪庤庼廋廙弇彧悢惇愔慥憖扅扊扺抃抔拃拤挦捯揳揾搒摽斝旴旸' \
                u'旻昉昽晞晳暅曈杧杻柈栒栟梽梾棁棨棬棻椑楯槚槜橦檵欸殣毐汈汭沄沘沚沨泃泚泜洑洣洨洴洺浉浐浕浡浥涘涢淏湉' \
                u'湜湝湨湲溇溠溦滃滍滘滪滫漹潏潟潵潽澥澴澼瀌瀍灈炟烜烝烺焌焜熜熥燊燏牁牂牚犨狉狝猄猇獏獴玕玙玚玠玡玥玦' \
                u'珣珰珽琇琎琤琫琯瑀瑄瑢璆璈璘璟璠璪瓘瓻甪畯疍疢疭瘆盉盫眊眬瞭矻砮硁硚碶礌礳祃祎祲祼祾禤秾穀穄窅窣窸竑' \
                u'筜筼箓篊篢簃簉簠糵繄纴纻绤绹罽羑翙翚翯耰耲脧腒膙臑臜芼茀茓茝荄荙莙菂菼萩葖蒟蓂蓇蕰藠蘘蜎螠螣蠋袆袗袪' \
                u'袯裈襕觱訄诐豨赑赟赪跐跶踶蹅蹐蹓蹜蹢蹽蹾轪辀辌辒邽郈郚郿鄌鄘酦醾銍鋆鍠钘铻锜锧镃镈镋镕镚镠镴镵闿阇阘' \
                u'靰靸靺靽靿鞁鞡鞧鞨韂韨颋颙飐飔飗饳饸饹饻馃馉馌骍骎骙鬹魆鱽鱾鲀鲃鲉鲊鲌鲏鲙鲪鲬鲯鲹鲾鳀鳁鳂鳈鳉鳑鳚鳡' \
                u'鳤鸤鸮鸰鸻鸼鹀鹐鹝鹟鹠鹡鹮鹲黇鼒鼩鼫鼱齁齉龁'
    chars_15_special_symbol = u'℅℉↖↗↘↙▽⊕㎎㎏㎜㎝㎞㎡㏄'

    chars = chars_6721 + chars_422 + chars_42_radical + chars_26_uppercase + chars_26_lowercase + chars_10_number + \
            chars_10_serial_number + chars_40_punctuation + chars_44_special_symbol + chars_15_special_symbol
    if not tranditional:
        chars = chars.replace(chars_422, u'')
        chars = chars.replace(chars_15_special_symbol, u'')
    if not radical:
        chars = chars.replace(chars_42_radical, u'')
    if not alphabeta:
        chars = chars.replace(chars_26_uppercase, u'')
        chars = chars.replace(chars_26_lowercase, u'')
    if not number:
        chars = chars.replace(chars_10_number, u'')
    if not serial_number:
        chars = chars.replace(chars_10_serial_number, u'')
    if not symbol:
        chars = chars.replace(chars_40_punctuation, u'')
        chars = chars.replace(chars_44_special_symbol, u'')
        chars = chars.replace(chars_15_special_symbol, u'')
    return chars

def arcurrency(chars):
    iunit = {u'零': 0, u'壹': 1, u'贰': 2, u'弍': 2, u'貳': 2, u'叁': 3, u'肆': 4, u'伍': 5, u'陆': 6, u'陸': 6, u'柒': 7, u'捌': 8,
             u'玖': 9, u'拾':10 , u'佰': 100, u'仟': 1000, u'万': 10000, u'萬': 10000, u'亿': 100000000, u'億': 100000000,
             u'元': 1 , u'圆': 1, u'整': 1, u'正': 1, u'角': 0.1, u'分': 0.01}
    split_chars_by_4_bit = u'亿億万萬元圆角分整正'
    split_chars_under_4_bit = u'仟佰拾'
    def split(chars, split_chars):
        idx = []
        for split_char in split_chars:
            idx.append(chars.find(split_char) + 1)
        idx.append(0)
        idx.append(len(chars))
        idx = list(set(idx))
        idx.sort()
        split_s = []
        for i in range(len(idx) - 1):
            split_s.append(chars[idx[i]:idx[i + 1]])
        return split_s
    def decode(chars):
        num = 0
        for char in chars:
            num = num * 10 + iunit[char]
        return num
    def decodeUnder4Bit(chars):
        print chars
        parts = split(chars=chars, split_chars=split_chars_under_4_bit)
        num = 0
        for part in parts:
            if part[-1] in split_chars_under_4_bit:
                if len(part) == 1:
                    num += iunit[part]
                elif len(part) == 2 and part[-2] == u'零':
                    num += iunit[part[-1]]
                else:
                    num += decode(chars=part[:-1]) * iunit[part[-1]]
            else:
                num += decode(chars=part)
        return num
    if len(chars) > 0 and chars[-1] not in split_chars_by_4_bit:
        add_tail = u'元整'
        if u'角' in chars:
            add_tail = u'分'
        elif u'元' in chars or u'圆' in chars:
            add_tail = u'分'
        chars += add_tail
    if u'元' not in chars and u'圆' not in chars:
        for idx in reversed(range(len(chars))):
            if chars[idx] in u'亿億万萬仟佰拾':
                chars = chars[:idx + 1] + u'元' + chars[idx + 1:]
                break
    parts = split(chars=chars, split_chars=split_chars_by_4_bit)
    num = 0
    for part in parts:
        if part[-1] in split_chars_by_4_bit:
            num += decodeUnder4Bit(chars=part[:-1]) * iunit[part[-1]]
        else:
            raise ValueError(u'金额格式有误:%s' % chars)
    return '%.2f' % num

def cncurrency(value, capital=False, prefix=u''):
    dunit = (u'角', u'分')
    if capital:
        word_2 = random.sample(u'弍貳', 1)[0]
        num = (u'零', u'壹', word_2, u'叁', u'肆', u'伍', u'陸', u'柒', u'捌', u'玖')
        iunit = [None, u'拾', u'佰', u'仟', u'萬', u'拾', u'佰', u'仟', u'億', u'拾', u'佰', u'仟', u'萬', u'拾', u'佰', u'仟']
    else:
        num = (u'零', u'壹', u'贰', u'叁', u'肆', u'伍', u'陆', u'柒', u'捌', u'玖')
        iunit = [None, u'拾', u'佰', u'仟', u'万', u'拾', u'佰', u'仟', u'亿', u'拾', u'佰', u'仟', u'万', u'拾', u'佰', u'仟']
        # num = (u'〇', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九')
        # iunit = [None, u'十', u'百', u'千', u'万', u'十', u'百', u'千', u'亿', u'十', u'百', u'千', u'万', u'十', u'百', u'千']
    iunit[0] = random.sample(u'元圆', 1)[0]
    if not isinstance(value, Decimal):
        value = Decimal(value).quantize(Decimal('0.01'))
    if value < 0:
        prefix += u'负'
        value = - value
    s = str(value)
    istr, dstr = s.split('.')
    istr = istr[::-1]
    so = []
    if value == 0:
        return prefix + num[0] + iunit[0]
    haszero = False
    if dstr == '00':
        haszero = True
    if dstr[1] != '0':
        so.append(dunit[1])
        so.append(num[int(dstr[1])])
    else:
        so.append(random.sample(u'整正', 1)[0])
    if dstr[0] != '0':
        so.append(dunit[0])
        so.append(num[int(dstr[0])])
    elif dstr[1] != '0':
        so.append(num[0])
        haszero = True
    if istr == '0':
        if haszero:
            so.pop()
        so.append(prefix)
        so.reverse()
        return ''.join(so)
    for i, n in enumerate(istr):
        n = int(n)
        if i % 4 == 0:
            if i == 8 and so[-1] == iunit[4]:
                so.pop()
            so.append(iunit[i])
            if n == 0:
                if not haszero:
                    so.insert(-1, num[0])
                    haszero = True
            else:
                so.append(num[n])
                haszero = False
        else:
            if n != 0:
                so.append(iunit[i])
                so.append(num[n])
                haszero = False
            else:
                if not haszero:
                    so.append(num[0])
                    haszero = True
    so.append(prefix)
    so.reverse()
    return ''.join(so)

def create_date(split=False, capital = True, character = True):
    year_dict = {'0': [u'零'],
                 '1': [u'壹'],
                 '2': [u'贰', u'貳', u'弍'],
                 '3': [u'叁'],
                 '4': [u'肆'],
                 '5': [u'伍'],
                 '6': [u'陆', u'陸'],
                 '7': [u'柒'],
                 '8': [u'捌'],
                 '9': [u'玖']}
    month_dict = {'01': [u'零壹', u'壹'],
                  '02': [u'零贰', u'零貳', u'零弍', u'贰', u'貳', u'弍'],
                  '03': [u'零叁', u'叁'],
                  '04': [u'零肆', u'肆'],
                  '05': [u'零伍', u'伍'],
                  '06': [u'零陆', u'零陸', u'陆', u'陸'],
                  '07': [u'零柒', u'柒'],
                  '08': [u'零捌', u'捌'],
                  '09': [u'零玖', u'玖'],
                  '10': [u'拾', u'壹拾', u'零拾', u'零壹拾'],
                  '11': [u'拾壹', u'壹拾壹', u'零壹拾壹'],
                  '12': [u'拾贰', u'拾貳', u'拾弍', u'壹拾贰', u'壹拾貳', u'壹拾弍', u'零壹拾贰', u'零壹拾貳', u'零壹拾弍']}
    day_dict = {'01': [u'零壹', u'壹'],
                '02': [u'零贰', u'零貳', u'零弍', u'贰', u'貳', u'弍'],
                '03': [u'零叁', u'叁'],
                '04': [u'零肆', u'肆'],
                '05': [u'零伍', u'伍'],
                '06': [u'零陆', u'零陸', u'陆', u'陸'],
                '07': [u'零柒', u'柒'],
                '08': [u'零捌', u'捌'],
                '09': [u'零玖', u'玖'],
                '10': [u'拾', u'壹拾', u'零拾', u'零壹拾'],
                '11': [u'拾壹', u'壹拾壹', u'零拾壹', u'零壹拾壹'],
                '12': [u'拾贰', u'拾貳', u'拾弍', u'壹拾贰', u'壹拾貳', u'壹拾弍', u'零拾贰', u'零拾貳', u'零拾弍', u'零壹拾贰', u'零壹拾貳', u'零壹拾弍'],
                '13': [u'拾叁', u'壹拾叁', u'零拾叁', u'零壹拾叁'],
                '14': [u'拾肆', u'壹拾肆', u'零拾肆', u'零壹拾肆'],
                '15': [u'拾伍', u'壹拾伍', u'零拾伍', u'零壹拾伍'],
                '16': [u'拾陆', u'拾陸', u'壹拾陆', u'壹拾陸', u'零拾陆', u'零拾陸', u'零壹拾陆', u'零壹拾陸'],
                '17': [u'拾柒', u'壹拾柒', u'零拾柒', u'零壹拾柒'],
                '18': [u'拾捌', u'壹拾捌', u'零拾捌', u'零壹拾捌'],
                '19': [u'拾玖', u'壹拾玖', u'零拾玖', u'零壹拾玖'],
                '20': [u'贰拾', u'貳拾', u'弍拾', u'零贰拾', u'零貳拾', u'零弍拾'],
                '21': [u'贰拾壹', u'貳拾壹', u'弍拾壹', u'零贰拾壹', u'零貳拾壹', u'零弍拾壹'],
                '22': [u'贰拾贰', u'貳拾貳', u'弍拾弍', u'零贰拾贰', u'零貳拾貳', u'零弍拾弍'],
                '23': [u'贰拾叁', u'貳拾叁', u'弍拾叁', u'零贰拾叁', u'零貳拾叁', u'零弍拾叁'],
                '24': [u'贰拾肆', u'貳拾肆', u'弍拾肆', u'零贰拾肆', u'零貳拾肆', u'零弍拾肆'],
                '25': [u'贰拾伍', u'貳拾伍', u'弍拾伍', u'零贰拾伍', u'零貳拾伍', u'零弍拾伍'],
                '26': [u'贰拾陆', u'貳拾陸', u'弍拾陸', u'零贰拾陆', u'零貳拾陸', u'零弍拾陸'],
                '27': [u'贰拾柒', u'貳拾柒', u'弍拾柒', u'零贰拾柒', u'零貳拾柒', u'零弍拾柒'],
                '28': [u'贰拾捌', u'貳拾捌', u'弍拾捌', u'零贰拾捌', u'零貳拾捌', u'零弍拾捌'],
                '29': [u'贰拾玖', u'貳拾玖', u'弍拾玖', u'零贰拾玖', u'零貳拾玖', u'零弍拾玖'],
                '30': [u'叁拾', u'零叁拾'],
                '31': [u'叁拾壹', u'零叁拾壹']}
    time_from = int(time.mktime((1970, 1, 1, 8, 0, 0, 0, 0, 0)))
    time_to = int(time.mktime((2100, 1, 1, 8, 0, 0, 0, 0, 0)))
    # time_from_most_1 = int(time.mktime((2017, 10, 20, 8, 0, 0, 0, 0, 0)))
    # time_to_most_1 = int(time.mktime((2017, 11, 15, 8, 0, 0, 0, 0, 0)))
    # time_from_most_2 = int(time.mktime((2017, 6, 1, 8, 0, 0, 0, 0, 0)))
    # time_to_most_2 = int(time.mktime((2018, 4, 1, 8, 0, 0, 0, 0, 0)))
    begin_time = time.time()
    chars_list = []
    while True:
        # random_range = random.randint(0, 9)
        # if random_range < 4:
        #     random_second = random.randint(time_from_most_1, time_to_most_1)
        # elif random_range < 8:
        #     random_second = random.randint(time_from_most_2, time_to_most_2)
        # else:
        random_second = random.randint(time_from, time_to)
        date_gen = time.localtime(random_second)
        # mode = random.randint(1, 1)
        if character:
            mode = 1
        else:
            mode = random.randint(2, 4)
        if mode == 1:
            year_chars = ''
            for char in '%04d' % date_gen.tm_year:
                way_write = year_dict[char]
                year_chars += random.sample(way_write, 1)[0]
            way_write = month_dict['%02d' % date_gen.tm_mon]
            month_chars = random.sample(way_write, 1)[0]
            way_write = day_dict['%02d' % date_gen.tm_mday]
            day_chars = random.sample(way_write, 1)[0]
            chars_tmp = year_chars + u'年' + month_chars + u'月' + day_chars + u'日'
            if not capital:
                chars_tmp = chars_tmp.replace(u'貳', u'贰')
                chars_tmp = chars_tmp.replace(u'弍', u'贰')
                chars_tmp = chars_tmp.replace(u'陸', u'陆')
        else:
            year_chars = '%04d' % date_gen.tm_year
            if mode == 2:
                month_chars = '%02d' % date_gen.tm_mon
                day_chars = '%02d' % date_gen.tm_mday
                chars_tmp = year_chars + month_chars + day_chars
            else:
                if random.randint(1, 2) == 1:
                    month_chars = '%02d' % date_gen.tm_mon
                    day_chars = '%02d' % date_gen.tm_mday
                else:
                    month_chars = '%d' % date_gen.tm_mon
                    day_chars = '%d' % date_gen.tm_mday
                if mode == 3:
                    chars_tmp = year_chars + '-' + month_chars + '-' + day_chars
                else:
                    chars_tmp = year_chars + '/' + month_chars + '/' + day_chars
        if split:
            if u'年' in chars_tmp:
                chars_part = chars_tmp.split(u'年')
                year_chars = chars_part[0]
                month_day_part = chars_part[1].split(u'月')
                month_chars = month_day_part[0]
                day_chars = month_day_part[1].replace(u'日', '')
            elif '/' in chars_tmp:
                chars_part = chars_tmp.split('/')
                year_chars = chars_part[0]
                month_chars = chars_part[1]
                day_chars = chars_part[2]
            elif '-' in chars_tmp:
                chars_part = chars_tmp.split('-')
                year_chars = chars_part[0]
                month_chars = chars_part[1]
                day_chars = chars_part[2]
            else:
                year_chars = chars_tmp[0:4]
                month_chars = chars_tmp[4:6]
                day_chars = chars_tmp[6:8]
            chars_tmp=[year_chars, month_chars, day_chars]
        yield chars_tmp

def random_choose(choice_list, num):
    chars = u''
    for i in range(num):
        chars += random.choice(choice_list)
    return chars

def character(chars = None, sample = 10, inf = 4, sup = 20, other_chars = u'', delete_chars = u''):
    all_chars = chars + other_chars
    for char in delete_chars:
        all_chars = all_chars.replace(char, u'')
    begin_time = time.time()
    chars_list = []
    print '正在生成汉字样本...'
    for loop in range(sample):
        chars_tmp = ''.join(char for char in random_choose(all_chars, random.randint(inf, sup)))
        chars_list.append(chars_tmp)
        progressbar(cur=loop + 1, total=sample, begin_time=begin_time, cur_time=time.time())
    return chars_list

def create_company_name(sample = 10, inf = 4, sup = 20):
    chars_list_org = character(sample=sample, inf=inf, sup=sup)
    end_chars = [u'有限公司', u'研究所', u'分行', u'支行', u'营业部', u'合作联社', u'']
    insert_chars = [u'(中国)', u'股份', u'企业', u'行']
    begin_time = time.time()
    chars_list = []
    print '正在生成公司名称样本...'
    for loop, chars_tmp in enumerate(chars_list_org):
        chars_length = len(chars_tmp)
        if random.randint(0, 3) == 0:
            if chars_length - 1 >= 1:
                idx = random.randint(1, chars_length - 1)
                chars_tmp = chars_tmp[0:idx] + random.sample(insert_chars, 1)[0] + chars_tmp[idx:]
        chars_tmp = chars_tmp + random.sample(end_chars, 1)[0]
        chars_list.append(chars_tmp)
        progressbar(cur=loop + 1, total=sample, begin_time=begin_time, cur_time=time.time())
    return chars_list

def create_digit_number(sample = 10, inf = 4, sup = 20, special = ''):
    all_chars = '0123456789'
    begin_time = time.time()
    chars_list = []
    print '正在生成数字样本...'
    for loop in range(sample):
        chars_tmp = ''
        while len(set(chars_tmp) - set(special)) == 0:
            chars_tmp = ''.join(char for char in random_choose(all_chars, random.randint(inf, sup)))
        if special:
            chars_tmp_list = list(chars_tmp)
            for i in range(random.randint(0, 4)):
                chars_tmp_list.insert(random.randint(0, len(chars_tmp_list)), random.sample(special, 1)[0])
            chars_tmp = ''.join(chars_tmp_list)
        chars_list.append(chars_tmp)
        progressbar(cur=loop + 1, total=sample, begin_time=begin_time, cur_time=time.time())
    return chars_list

def create_amount_in_figure(inf = 4, sup = 13, prefix=u'￥'):
    all_chars = u'0123456789'
    chars_list = []
    #print '正在生成小写金额样本...'
    while True:
        decimal_part = random_choose(all_chars, 2)
        integer_part = str(int(random_choose(all_chars, random.randint(inf - 3, sup - 3))))
        yield u'%s%s.%s' % (prefix, integer_part, decimal_part)


def create_amount_in_word(inf = 4, sup = 13, capital = False):
    for chars_tmp in create_amount_in_figure(inf=inf, sup=sup, prefix=u''):
        chars_tmp=u'890000000000.21'
        yield cncurrency(value=chars_tmp, capital=capital)
