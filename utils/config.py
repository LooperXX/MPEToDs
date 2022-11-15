import argparse, logging, datetime, os, json, fitlog, random, time
import numpy as np
from tqdm import tqdm

# Pre-Define Lang
UNK_token = 0
PAD_token = 1
EOS_token = 2
SOS_token = 3

EOS = '<|endoftext|>'
USR = '<|USR|>'
SYS = '<|SYS|>'


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt='%Y/%d/%m %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


parser = argparse.ArgumentParser(description='MPEToDs')
# Dataset and Other Parameters
parser.add_argument('-ds', '--dataset', help='dataset name', required=False, default="smd")
parser.add_argument('-op', '--output_dir', help='output path', required=False)
parser.add_argument('--gpu', '-g', action='store_true', help='use gpu', required=False, default=False)
parser.add_argument('-random_seed', '--random_seed', help='random_seed', type=int, required=False, default=42)
parser.add_argument('-mxl', '--max_line', help='load file max line for debugging', required=False, type=int,
                    default=None)
parser.add_argument('-uf', '--use_fitlog', help='use fitlog', action='store_true', required=False, default=False)
parser.add_argument('-ft', '--fine_tune', help='fine_tune', action='store_true', required=False, default=False)
parser.add_argument('-path', '--path', help='path of the file to load', required=False, default=None)
parser.add_argument('-ppath', '--pretrain_path', help='path of the pre-train file to load', required=False,
                    default='pre-train/')
parser.add_argument('-pname', '--pretrain_name', help='name of the pre-train file to load', required=False,
                    default='medium_ft.pkl')
parser.add_argument('-pgpt', '--pretrain_GPT', help='load pre-train GPT', required=False,
                    default=None)
parser.add_argument('-pkb', '--pretrain_KB', help='load pretrain KB', required=False,
                    default=None)
parser.add_argument('-daf', '--data_augmentation_file', help='data augmentation file', required=False,
                    default='train_10.txt')
parser.add_argument('-fg', '--freeze_GPT', help='freeze GPT2 model', action='store_true', required=False, default=False)
parser.add_argument('--fp16', action='store_true', required=False, default=False)
parser.add_argument('-fld', '--fit_log_dir', required=False, default='logs/')

# Model Parameters
parser.add_argument('-hop', '--hop', help='Hop Number', required=False, type=int, default=3)
parser.add_argument('-hdd', '--hidden', help='Hidden size for Memory Network', required=False, type=int, default=128)
parser.add_argument('-em', '--embedding_dim', help='embedding dim', required=False, type=int, default=1024)
parser.add_argument('-dr', '--drop', help='Drop Out for Memory Network', required=False, default=0.2, type=float)
parser.add_argument('-abg', '--ablationG', help='ablation global memory pointer', type=int, required=False, default=0)
parser.add_argument('-abh', '--ablationH', help='ablation context embedding', type=int, required=False, default=1)

# Training Parameters
parser.add_argument('-bsz', '--batch', help='batch size', required=False, type=int, default=4)
parser.add_argument('-accs', '--accumulation_steps', type=int, default=2,
                    help="to increase effective batch size and reduce synchronization")
parser.add_argument('-lr', '--learn', help='learning rate', required=False, default=0.001, type=float)
parser.add_argument('-gl', '--GPT2_learn', help='Generation Module learning rate', required=False, default=0.00001, type=float)
parser.add_argument('-clip', '--clip', help='gradient clipping', required=False, type=int, default=1)
parser.add_argument('-m', '--metric', help='Early Stop Metric, BLEU or ENTF1', required=False, default='ENTF1')
parser.add_argument('-rec', '--record', help='use record function during inference', type=int, required=False,
                    default=1)

parser.add_argument('-pa', '--patience', help='patience for early stop in pretrain', required=False, type=int,
                    default=8)
parser.add_argument('-evalp', '--evalp', help='evaluation period', required=False, type=int, default=1)
parser.add_argument("--valid_steps", type=int, default=2000, help="how many optim steps between validations")
parser.add_argument("--warmup_steps", type=int, default=2000)
parser.add_argument("--logging_steps", type=int, default=200)
parser.add_argument("--total_steps", help='max training steps for pretraining', type=int, default=200000)

args = vars(parser.parse_args())

if args['dataset'] == 'smd':
    MEM_TOKEN_SIZE = 6
elif args['dataset'] == 'woz':
    MEM_TOKEN_SIZE = 12
elif args['dataset'] == 'cam':
    MEM_TOKEN_SIZE = 11
timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')

if not args['use_fitlog']:
    fitlog.debug()

# pretrain GPT2
if args['pretrain_GPT'] is None and args['pretrain_KB'] is None and not args['fine_tune']:
    args['output_dir'] = os.path.join('save', 'pretrain_GPT_{}_{}_{}_{}_{}_{}'.
                                      format(args['pretrain_name'],
                                             args['batch'],
                                             args['accumulation_steps'],
                                             args['GPT2_learn'],
                                             args['warmup_steps'],
                                             timestamp))
# pretrain KB
elif args['pretrain_GPT'] is not None and args['pretrain_KB'] is None and not args['fine_tune']:
    args['output_dir'] = os.path.join('save', 'pretrain_KB_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.
                                      format(args['pretrain_name'],
                                             args['dataset'],
                                             args['batch'],
                                             args['accumulation_steps'],
                                             args['drop'],
                                             args['hidden'],
                                             args['learn'],
                                             args['GPT2_learn'],
                                             args['warmup_steps'],
                                             timestamp))
# fine tune
elif args['pretrain_GPT'] is not None and args['pretrain_KB'] is not None and args['fine_tune']:
    args['output_dir'] = os.path.join('save', 'fine_tune_{}_{}_{}_{}_{}_{}_{}_{}_{}'.
                                      format(args['pretrain_name'],
                                             args['dataset'],
                                             args['batch'],
                                             args['accumulation_steps'],
                                             args['drop'],
                                             args['learn'],
                                             args['GPT2_learn'],
                                             args['warmup_steps'],
                                             timestamp))
else:
    print('config error!')
    exit(1)

os.makedirs(args['output_dir'], exist_ok=True)
log_path = os.path.join(args['output_dir'], "config.json")
with open(log_path, "w", encoding="utf8") as fw:
    fw.write(json.dumps(args))

mylogger = get_logger(os.path.join(args['output_dir'], 'log.txt'), name='MPEToDs')
mylogger.info(str(args))
mylogger.info('train batch size = {}, new train batch size (after gradient accumulation) = {}'.format(
    args['batch'], args['batch'] * args['accumulation_steps']))

mylogger.info("16-bits training: {}".format(args['fp16']))

# load data process function
metric = args['metric']
if args['pretrain_GPT'] is None and args['pretrain_KB'] is None and not args['fine_tune']:
    from utils.utils_pretrain import *
elif args['dataset'] == 'smd':
    head_test = 'epoch,acc,bleu,f1_macro,f1_micro,f1_macro_sche,f1_macro_wea,f1_macro_nav,f1_micro_sche,f1_micro_wea,f1_micro_nav'
    from utils.utils_Ent_smd import *
elif args['dataset'] == 'woz':
    head_test = 'epoch,acc,bleu,f1_macro,f1_micro,f1_macro_res,f1_macro_attr,f1_macro_hotel,f1_micro_res,f1_micro_attr,f1_micro_hotel'
    from utils.utils_Ent_woz import *
elif args['dataset'] == 'cam':
    head_test = 'epoch,acc,bleu,f1_macro,f1_micro'
    from utils.utils_Ent_cam import *
else:
    mylogger.error("[ERROR] You need to provide the --dataset information")

# load loss and metric logger
if args['fine_tune']:
    eval_logger = open(os.path.join(args['output_dir'], 'eval_log.txt'), 'a+')
    print(head_test, file=eval_logger)
    test_logger = open(os.path.join(args['output_dir'], 'test_log.txt'), 'a+')
    print(head_test, file=test_logger)

# fixed random seed
torch.manual_seed(args['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    torch.backends.cudnn.deterministic = True
np.random.seed(args['random_seed'])
random.seed(args['random_seed'])

# logger
os.makedirs(args['fit_log_dir'], exist_ok=True)
fitlog.set_log_dir(args['fit_log_dir'])
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)

# special tokens

ent_types = ['@where_to@', '@hotel2_detail.name.hotel@', '@flight_booked.flight_booked@', '@auto_repair.name.store@',
             '@mls.name.player@', '@from_location@', '@flight3_detail.date@', '@hotel4_detail.amenity@',
             '@flight3_detail.stops@', '@movie_search.name.theater@', '@alarm_time@', '@movie_search.type.screening@',
             '@outbound_arrival_time@', '@flight_search.total_fare@', '@hotel1_detail.price_per_night@',
             '@hotel4_detail.star_rating@', '@flight3_detail.from@', '@flight1_detail.date@', '@venue_address@',
             '@ticket@', '@flight1_detail.seat_location@', '@hotel_search.name.hotel@',
             '@hotel3_detail.customer_review@', '@flight_search.seat_location@', '@flight3_detail.stops.location@',
             '@show_time@', '@nba.result.match@', '@hotel_search.date_range@', '@restaurant.sub-location@', '@fee@',
             '@show_date@', '@flight4_detail.airline@', '@mlb.other_description.person@', '@mls.day.match@',
             '@mls.position.player@', '@movie_search.other_description@', '@flight1_detail.airline@',
             '@movie_search.runtime@', '@mls.other_description.person@', '@dest@', '@precipitation@',
             '@mlb.record.team@', '@movie_ticket.ticket_booking@', '@flight_search.stops@', '@review.audience@',
             '@music.type.music@', '@epl.name.non_player@', '@hotel_search.star_rating@', '@moviename@',
             '@mlb.place.team@', '@auto_repair.auto_repair@', '@address@', '@ride_fare@', '@average_rating@',
             '@from_station@', '@epl.record.player@', '@nba.name.non_player@', '@aggregate_rating@',
             '@pizza_ordering.type.topping@', '@duration.movie@', '@category@', '@restaurant_reservation.type.seating@',
             '@nba.record.games_back@', '@cuisine@', '@nfl.record.player@', '@epl.record.team@',
             '@restaurant_reservation.name.restaurant@', '@epl.score.match@', '@mls.score.match@',
             '@mls.record.player@', '@auto_repair.year.vehicle@', '@city@', '@destination@', '@street_address@',
             '@pickup_city@', '@movie_ticket.time.end@', '@date.release@', '@food_order.location.restaurant@',
             '@nba.record.player@', '@pickup_time@', '@mlb.other_description.team@', '@actor@',
             '@hotel4_detail.name.hotel@', '@price.total@', '@pickup_location_city@', '@mls.record.games_back@',
             '@hotel_search.customer_rating@', '@origin_city@', '@hotel4_detail.total_fare@',
             '@flight4_detail.other_description@', '@artist@', '@flight4_detail.from.time@', '@restaurant.phone@',
             '@mlb.name.player@', '@recipient_account_name@', '@price.ticket@', '@movie_name@', '@subcategory@',
             '@flight_search.destination2@', '@music.technical_difficulty@', '@movie_search.release_date@',
             '@hotel2_detail.amenity@', '@song_name@', '@departure_time@', '@food_order.total_price@',
             '@mls.time.match@', '@mlb.venue@', '@hotel3_detail.other_detail@', '@mlb.score.match@', '@epl.place.team@',
             '@uber_lyft.price.estimate@', '@movie_ticket.seating@', '@food_order.name.item@', '@nfl.result.match@',
             '@flight4_detail.fare@', '@nfl.position.player@', '@epl.record.games_ahead@', '@music.describes_genre@',
             '@food_order.name.restaurant@', '@inbound_departure_time@', '@flight2_detail.seat_location@',
             '@weekly_time@', '@dropoff_location@', '@flight2_detail.date@', '@location@', '@music.name.track@',
             '@flight4_detail.to@', '@origin_airport@', '@time.showing@', '@nfl.other_description.match@',
             '@starttime@', '@flight_search.time_of_day@', '@transfer_time@', '@pizza_ordering.pizza_ordering@',
             '@transfer_amount@', '@hotel_search.check-in_date@', '@restaurant_name@', '@movie_search.price.streaming@',
             '@mlb.date.match@', '@name.theater@', '@people@', '@leaving_date@', '@area@', '@destination_airport_name@',
             '@auto_repair.time.appt@', '@movie_search.critic_rating@', '@amount@', '@movie_ticket.name.theater@',
             '@hotel2_detail.location@', '@mls.name.team@', '@pizza_ordering.preference@', '@mlb.day.match@',
             '@hotel_booked.hotel_booked@', '@music.describes_artist@', '@flight_search.num.pax@',
             '@flight2_detail.airline@', '@destination_airport@', '@epl.other_description.match@',
             '@address_of_location@', '@flight2_detail.other_description@', '@nba.other_description.match@',
             '@distanceconstraints@', '@hotel1_detail.location@', '@mls.place.team@', '@nfl.other_description.person@',
             '@taskcomplete@', '@implicit_value@', '@food_order.type.food@', '@venue@', '@event@',
             '@nfl.record.games_back@', '@restaurantname@', '@coffee_ordering.type.milk@', '@food@', '@leaving_time@',
             '@flight4_detail.from@', '@flight1_detail.other_description@', '@restaurant.location@',
             '@approximate_ride_duration@', '@flight1_detail.flight_number@', '@date@',
             '@hotel4_detail.price_per_night@', '@visit_date@', '@traffic_info@', '@movie_ticket.location.theater@',
             '@rent@', '@hotel_search.location.hotel@', '@flight1_detail.seating_class@',
             '@music.describes_type.music@', '@mlb.name.non_player@', '@hotel4_detail.customer_rating@',
             '@movie_ticket.price.ticket@', '@flight1_detail.from.time@', '@flight2_detail.to.time@',
             '@destination_city@', '@uber_lyft.uber_lyft@', '@flight_search.origin@', '@time.preference@',
             '@pizza_ordering.name.store@', '@mc_list@', '@rating@', '@event_date@', '@type@',
             '@movie_ticket.time.start@', '@flight1_detail.stops@', '@nba.record.team@', '@nfl.time.match@',
             '@auto_repair.date.appt@', '@choice@', '@auto_repair.reason.appt@', '@car_name@',
             '@hotel_search.check-out_date@', '@flight4_detail.seating_class@', '@hotel3_detail.type.room@',
             '@flight3_detail.fare@', '@mls.other_description.team@', '@flight3_detail.to.time@',
             '@uber_lyft.num.people@', '@personfullname@', '@atmosphere@', '@mpaa_rating@', '@available_end_time@',
             '@hotel3_detail.name.hotel@', '@hotel2_detail.type.room@', '@hotel3_detail.amenity@',
             '@mls.name.non_player@', '@music.name.artist@', '@hotel2_detail.other_detail@',
             '@coffee_ordering.num.drink@', '@hotel4_detail.location@', '@epl.day.match@', '@ref@',
             '@restaurant.menu_item@', '@restaurant_reservation.name.reservation@',
             '@restaurant_reservation.restaurant_reservation@', '@hotel_search.sub_location.hotel@',
             '@flight1_detail.fare@', '@flight_search.type@', '@phonenumber@', '@coffee_ordering.coffee_order@',
             '@arrive@', '@number_of_days@', '@zip@', '@flight2_detail.from.time@', '@music.name.album@',
             '@rating.movie@', '@temperature@', '@flight_search.price_range@', '@departure_date@',
             '@hotel4_detail.type.room@', '@genre@', '@flight4_detail.stops@', '@restaurant.offical_description@',
             '@outbound_departure_time@', '@seating@', '@hotel_search.type.room@', '@theater_chain@', '@room@',
             '@description@', '@return_date@', '@flight3_detail.to@', '@day@', '@hotel3_detail.total_fare@',
             '@flight_search.luggage@', '@flight1_detail.to@', '@hotel2_detail.customer_review@',
             '@food_order.other_description.item@', '@flight2_detail.to@', '@restaurant_reservation.num.guests@',
             '@theater_name@', '@new_alarm_time@', '@hotel_search.num.guests@', '@name.genre@', '@wait_time@',
             '@mls.date.match@', '@cost@', '@nba.position.player@', '@epl.name.team@', '@hotel_search.other_request@',
             '@epl.result.match@', '@pricerange@', '@recipient_name@', '@pizza_ordering.size.pizza@',
             '@movie_search.movie_rating@', '@uber_lyft.time.pickup@', '@name@', '@nfl.date.match@',
             '@flight2_detail.stops.location@', '@theater@', '@restauranttype@', '@movie_search.price.ticket@',
             '@nba.name.team@', '@epl.venue@', '@epl.date.match@', '@time@', '@hotel1_detail.other_detail@',
             '@nfl.day.match@', '@restaurant_reservation.reservation@', '@postcode@', '@nba.score.match@',
             '@mls.record.team@', '@depart@', '@poi@', '@uber_lyft.location.from@', '@movie_search.location.theater@',
             '@agenda@', '@food_order.num.people@', '@pizza_ordering.pizza_order@', '@nba.date.match@',
             '@nba.time.match@', '@movie_ticket.name.movie@', '@doctor_name@', '@origin_airport_name@',
             '@nfl.other_description.team@', '@auto_repair.name.customer@', '@actress@', '@flight3_detail.airline@',
             '@therapist_name@', '@food_order.official_description.restaurant@', '@hotel1_detail.total_fare@', '@car@',
             '@fare@', '@state@', '@mls.record.games_ahead@', '@flight2_detail.fare@', '@auto_repair.name.vehicle@',
             '@hotel3_detail.star_rating@', '@weather_attribute@', '@nfl.venue@', '@flight4_detail.to.time@',
             '@pizza_ordering.location.store@', '@movie_search.synopsis@', '@movie_search.time.start@', '@phone@',
             '@directed_by@', '@food_order.rating.restaurant@', '@epl.name.player@',
             '@coffee_ordering.coffee_ordering@', '@available_start_time@', '@coffee_ordering.location.store@',
             '@nba.other_description.person@', '@nfl.name.team@', '@dress_code@', '@description.plot@',
             '@movie_search.genre@', '@restaurant_reservation.location.restaurant@', '@flight_search.airline@',
             '@hotel_search.num.rooms@', '@phone_number@', '@nfl.record.games_ahead@', '@num.tickets@',
             '@dentist_name@', '@flight2_detail.seating_class@', '@stay_length@', '@nba.venue@',
             '@coffee_ordering.size.drink@', '@mlb.name.team@', '@movie_ticket.num.tickets@',
             '@flight_search.date.depart_intermediate@', '@mlb.record.games_back@', '@nba.day.match@',
             '@movie_search.audience_rating@', '@mlb.other_description.match@', '@hotel_name@', '@to_location@',
             '@movie_ticket.movie_ticket@', '@mlb.record.player@', '@pizza_ordering.type.crust@',
             '@restaurant.price_range@', '@new_alarm_name@', '@restaurant.rating@', '@uber_lyft.time.dropoff@',
             '@movie_search.streaming_service@', '@hotel_search.price_range@', '@flight2_detail.from@', '@party@',
             '@city_of_event@', '@critic_rating@', '@property_name@', '@epl.position.player@',
             '@epl.other_description.person@', '@inbound_arrival_time@', '@hotel_search.num.beds@', '@stylist_name@',
             '@music.describes_track@', '@humidity@', '@price_per_night@', '@movie_ticket.time.duration@',
             '@restaurant.type.meal@', '@stay@', '@food_order.type.retrieval@', '@to_station@',
             '@flight3_detail.flight_number@', '@destination_station_name@', '@auto_repair.appointment@',
             '@nba.other_description.team@', '@flight1_detail.stops.location@', '@mls.other_description.match@',
             '@hotel3_detail.price_per_night@', '@movie_search.real_person@', '@food_order.type.meal@', '@pricing@',
             '@flight3_detail.other_description@', '@hotel1_detail.amenity@', '@movie_search.name.movie@',
             '@food_order.time.pickup@', '@total_price@', '@uber_lyft.type.ride@', '@nba.place.team@', '@album@',
             '@balance@', '@mealtype@', '@epl.time.match@', '@place_name@', '@music.describes_album@',
             '@nfl.score.match@', '@numberofpeople@', '@price@', '@hotel1_detail.type.room@', '@nfl.place.team@',
             '@auto_repair.location.store@', '@epl.other_description.team@', '@flight2_detail.stops@',
             '@hotel3_detail.customer_rating@', '@origin@', '@appointment_date@', '@hotel2_detail.total_fare@',
             '@uber_lyft.location.to@', '@flight_search.date.depart_origin@', '@title@', '@hotel_search.amenity@',
             '@flight1_detail.to.time@', '@restaurant.type.food@', '@actors@', '@hotel1_detail.customer_review@',
             '@mls.result.match@', '@distance@', '@nfl.name.non_player@', '@dropoff_location_city@',
             '@nba.name.player@', '@restaurant.business_hours@', '@movie_search.character@', '@movie_search.time.end.@',
             '@flight_search.destination1@', '@department@', '@multiple_choice@', '@uber_lyft.duration.estimate@',
             '@flight2_detail.flight_number@', '@mlb.time.match@', '@flight3_detail.seat_location@',
             '@nfl.record.team@', '@wind@', '@hotel1_detail.name.hotel@', '@movie_series@', '@attraction_name@',
             '@starring@', '@leave@', '@review.critic@', '@director@', '@event_name@',
             '@flight4_detail.stops.location@', '@restaurant.name.restaurant@', '@pickup_location@', '@pickup_date@',
             '@restaurant_reservation.time.reservation@', '@id@', '@hotel1_detail.customer_rating@',
             '@description.other@', '@video_format@', '@uber_lyft.ride_booking@', '@date.showing@',
             '@flight4_detail.date@', '@restaurant.other_description@', '@mls.venue@', '@pizza_ordering.name.pizza@',
             '@coffee_ordering.name.drink@', '@account_balance@', '@numberofkids@', '@movie_ticket.type.screening@',
             '@music.name.genre@', '@flight1_detail.from@', '@name.movie@', '@hotel4_detail.other_detail@',
             '@dropoff_date@', '@mlb.position.player@', '@hotel2_detail.customer_rating@',
             '@hotel2_detail.star_rating@', '@name.character@', '@flight_search.other_description@',
             '@nfl.name.player@', '@open@', '@check_out_date@', '@origin_station_name@', '@hotel3_detail.location@',
             '@flight3_detail.seating_class@', '@mlb.result.match@', '@name.person@', '@alarm_name@',
             '@flight_search.date.return@', '@hotel_search.type.bed@', '@car_type@', '@mlb.record.games_ahead@',
             '@event_time@', '@flight3_detail.from.time@', '@occasion@', '@check_in_date@', '@type.screening@',
             '@flight_search.seating_class@', '@epl.record.games_back@', '@food_order.price_range@',
             '@appointment_time@', '@coffee_ordering.preference@', '@hotel2_detail.price_per_night@',
             '@hotel1_detail.star_rating@', '@event_location@', '@stars@', '@restaurant_reservation.date.reservation@']
